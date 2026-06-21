"""Loads a Meta DINOv2 teacher checkpoint via DINOv2's own config + build path."""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger(__name__)

_DEFAULT_NORM_MEAN = (0.485, 0.456, 0.406)  # ImageNet (DINOv2 default)
_DEFAULT_NORM_STD = (0.229, 0.224, 0.225)


def _add_submodule_to_path(submodule_path: Path) -> None:
    submodule_path = Path(submodule_path).resolve()
    if not submodule_path.exists():
        raise FileNotFoundError(f"dinov2 submodule not found at {submodule_path}")
    if str(submodule_path) not in sys.path:
        sys.path.insert(0, str(submodule_path))


class LocalDinoV2Extractor:
    """Loads a Meta DINOv2 teacher checkpoint and exposes `extract_tokens()`."""

    def __init__(
        self,
        config_file: str,
        pretrained_weights: str,
        submodule_path: str,
        opts: list[str] | None = None,
        norm_mean: tuple[float, float, float] = _DEFAULT_NORM_MEAN,
        norm_std: tuple[float, float, float] = _DEFAULT_NORM_STD,
    ):
        _add_submodule_to_path(Path(submodule_path))

        from omegaconf import OmegaConf
        from dinov2.configs import dinov2_default_config
        from dinov2.models import build_model_from_cfg
        from dinov2.utils.utils import load_pretrained_weights

        default_cfg = OmegaConf.create(dinov2_default_config)
        file_cfg = OmegaConf.load(config_file)
        cli_cfg = OmegaConf.from_cli(opts) if opts else OmegaConf.create({})
        cfg = OmegaConf.merge(default_cfg, file_cfg, cli_cfg)

        LOGGER.info(f"Loading local DINOv2 from {pretrained_weights}")
        LOGGER.info(f"  config_file: {config_file}")
        if opts:
            LOGGER.info(f"  opts: {opts}")

        model, _ = build_model_from_cfg(cfg, only_teacher=True)
        load_pretrained_weights(model, pretrained_weights, "teacher")
        model.eval().to(DEVICE)

        self.model = model
        self.embed_dim: int = model.embed_dim
        self.patch_size: int = cfg.student.patch_size
        self.num_register_tokens: int = cfg.student.num_register_tokens
        self._norm_mean = torch.tensor(norm_mean).view(1, 3, 1, 1).to(DEVICE)
        self._norm_std = torch.tensor(norm_std).view(1, 3, 1, 1).to(DEVICE)
        LOGGER.info(
            f"Local DINOv2 ready: embed_dim={self.embed_dim} "
            f"patch_size={self.patch_size} num_register_tokens={self.num_register_tokens} "
            f"norm_mean={norm_mean} norm_std={norm_std}"
        )

    def _preprocess(self, pil_imgs: list, res: int) -> torch.Tensor:
        if res % self.patch_size != 0:
            raise ValueError(f"res={res} not divisible by patch_size={self.patch_size}")
        tensors = []
        for img in pil_imgs:
            img = img.resize((res, res), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        batch = torch.stack(tensors, dim=0).to(DEVICE)
        batch = (batch - self._norm_mean) / self._norm_std
        return batch

    @torch.no_grad()
    def extract_tokens(
        self,
        pil_imgs: list,
        res: int,
        layers: list,
        agg_method: str,
        grouped_layers: list | None = None,
        docrop: bool = False,
        use_clahe: bool = False,
        dino_saliency_layer: int = 0,
        token_type: str = "patch",
    ):
        """Returns (tokens [B,h_p,w_p,C], (h_p,w_p), saliency).

        token_type: "patch" → spatial grid; "cls" → [B, 1, 1, C].
        """
        del grouped_layers, docrop, use_clahe, dino_saliency_layer

        x = self._preprocess(pil_imgs, res)
        B = x.shape[0]

        depth = (
            len(self.model.blocks[-1]) if self.model.chunked_blocks
            else len(self.model.blocks)
        )
        # Convert negatives to positives, preserving user order.
        positives_user_order = list(dict.fromkeys(
            (li if li >= 0 else depth + li) for li in layers
        ))
        # DINOv2 returns outputs in ascending block-index order regardless of
        # the order in `n`. Sort for the API call, then permute back so concat
        # aggregation preserves the user's intended layer ordering.
        positives_sorted = sorted(positives_user_order)
        sorted_index = {p: i for i, p in enumerate(positives_sorted)}
        user_to_sorted = [sorted_index[p] for p in positives_user_order]

        if token_type == "cls":
            outs = self.model.get_intermediate_layers(
                x, n=positives_sorted, reshape=False, return_class_token=True, norm=True
            )
            # outs is in sorted order; reorder to user-given order.
            cls_per_layer = [outs[i][1] for i in user_to_sorted]  # each [B, D]
            if agg_method == "mean":
                fused = torch.stack(cls_per_layer, dim=0).mean(dim=0)  # [B, D]
            elif agg_method == "concat":
                fused = torch.cat(cls_per_layer, dim=-1)               # [B, D*n_layers]
            else:
                raise ValueError(f"agg_method={agg_method} not supported (mean/concat only)")
            fused = fused.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
            return fused.cpu().numpy(), (1, 1), np.zeros((B, 1, 1), dtype=np.float32)

        # token_type == "patch"
        h_p = w_p = res // self.patch_size
        outs = self.model.get_intermediate_layers(
            x, n=positives_sorted, reshape=False, return_class_token=False, norm=True
        )
        # Reorder outs from sorted → user order before reshape.
        outs = [outs[i] for i in user_to_sorted]
        spatial = []
        for o in outs:
            if o.shape[1] != h_p * w_p:
                raise RuntimeError(
                    f"unexpected token count {o.shape[1]} (expected {h_p*w_p})"
                )
            spatial.append(o.reshape(B, h_p, w_p, -1))

        if agg_method == "mean":
            fused = torch.stack(spatial, dim=0).mean(dim=0)
        elif agg_method == "concat":
            fused = torch.cat(spatial, dim=-1)
        else:
            raise ValueError(f"agg_method={agg_method} not supported (mean/concat only)")

        return fused.cpu().numpy(), (h_p, w_p), np.zeros((B, h_p, w_p), dtype=np.float32)
