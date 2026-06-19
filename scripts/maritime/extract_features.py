"""Incremental DINOv2 feature extraction.

Called once per in-dist subset (100, 500, 1000, 5000, 10000 perCat). On each
call, loads any existing cache, identifies which images of the requested
subset are NEW (not yet in the cache), runs DINOv2 on only the new ones,
appends them, saves `subset_idx_{size}` for the current --size, carries
prior `subset_idx_*` entries forward, and resaves.

Workflow (per dataset):
    extract --in_dist_subset In-distribution_100perCat --out_cache_file rgb.npz
    extract --in_dist_subset In-distribution_500perCat --out_cache_file rgb.npz
    extract --in_dist_subset In-distribution_1000perCat --out_cache_file rgb.npz
    extract --in_dist_subset In-distribution_5000perCat --out_cache_file rgb.npz
    extract --in_dist_subset In-distribution_10000perCat --out_cache_file rgb.npz

Each invocation only does the DINOv2 forward pass for images NOT already in
the cache. val/test features are extracted once (on first call) and reused.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from subspacead.core.extractor_local import LocalDinoV2Extractor
from subspacead.data.maritime import list_fit_paths, list_test_paths, resolve_roots

import _common
from _common import _sig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract ALL 24 transformer layers so downstream anomaly_detection.py can
# pick any subset without re-running the (expensive) DINOv2 forward.
LAYER_UNION = list(range(-24, 0))  # -1, -2, ..., -24


def local_forward(extractor: LocalDinoV2Extractor, positives_sorted, image_res):
    """Build a forward_batch(pils) -> Tensor[B, depth, D] for the local backbone.

    Preprocesses the PIL batch at `image_res` and runs get_intermediate_layers,
    stacking the per-layer class tokens.
    """
    @torch.no_grad()
    def forward_batch(pil_imgs):
        x = extractor._preprocess(pil_imgs, image_res)
        outs = extractor.model.get_intermediate_layers(
            x, n=positives_sorted, reshape=False,
            return_class_token=True, norm=True,
        )
        return torch.stack([c for (_p, c) in outs], dim=1)
    return forward_batch


def extract_per_layer_cls(
    extractor: LocalDinoV2Extractor,
    paths: list[str],
    batch_size: int,
    image_res: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    depth = (
        len(extractor.model.blocks[-1]) if extractor.model.chunked_blocks
        else len(extractor.model.blocks)
    )
    positives_sorted = sorted({(li if li >= 0 else depth + li) for li in LAYER_UNION})
    cls_all, valid, _secs, _n = _common.extract_per_layer_cls(
        paths, batch_size, image_res, desc,
        local_forward(extractor, positives_sorted, image_res),
    )
    return cls_all, valid


def main():
    p = argparse.ArgumentParser(description="Incremental DINOv2 feature extraction.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--in_dist_subset", required=True,
                   help="Subset to extract THIS round (e.g. In-distribution_100perCat). "
                        "Only images NEW to the cache will trigger DINOv2 forwards.")
    p.add_argument("--size", type=int, required=True,
                   help="Subset size for THIS round. The saved index key will be "
                        "named 'subset_idx_{size}'. Previously-saved subset_idx_* "
                        "entries in the cache are preserved.")
    p.add_argument("--id_test_subset", required=True)
    p.add_argument("--ood_subset", default="Out-of-distribution")
    p.add_argument("--config_file", required=True)
    p.add_argument("--pretrained_weights", required=True)
    p.add_argument("--local_submodule_path", required=True)
    p.add_argument("--local_norm_mean", type=float, nargs=3, required=True)
    p.add_argument("--local_norm_std", type=float, nargs=3, required=True)
    p.add_argument("--image_res", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out_cache_file", type=Path, required=True,
                   help="cumulative cache .npz; created if missing, appended otherwise")
    p.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format=f"[extract] %(asctime)s %(message)s")

    # === Resolve subset paths ===
    in_dist_root, ood_root = resolve_roots(
        Path(args.data_root), args.in_dist_subset, args.ood_subset
    )
    id_test_root = Path(args.data_root) / args.id_test_subset
    if not id_test_root.exists():
        raise FileNotFoundError(f"id_test_root not found: {id_test_root}")
    requested_paths = list_fit_paths(in_dist_root, seed=42)

    # === Load existing cache (if any) ===
    cache_exists = args.out_cache_file.exists()
    prior_subset_idx: dict = {}
    if cache_exists:
        d = np.load(args.out_cache_file, allow_pickle=False)
        cached_fit_cls   = d["fit_cls"]
        cached_fit_paths = list(d["fit_paths"].astype(str))
        val_cls  = d["val_cls"]
        test_cls = d["test_cls"]
        val_valid  = d["val_valid"]
        test_valid = d["test_valid"]
        val_labels  = d["val_labels"]
        test_labels = d["test_labels"]
        depth = int(d["depth"])
        positives_sorted = d["positives_sorted"].tolist()
        cached_image_res = int(d["image_res"])
        if cached_image_res != args.image_res:
            raise RuntimeError(
                f"cache image_res ({cached_image_res}) != requested ({args.image_res}); "
                f"normalization/extraction args must match across invocations"
            )
        prior_subset_idx = {
            k: d[k] for k in d.files if k.startswith("subset_idx_")
        }
        cached_sigs = {_sig(p): i for i, p in enumerate(cached_fit_paths)}
        logging.info(
            f"loaded cache: {len(cached_fit_paths)} fit images, "
            f"val={val_cls.shape}, test={test_cls.shape}, "
            f"prior subset indices: {sorted(prior_subset_idx.keys())}"
        )
    else:
        cached_fit_cls = None
        cached_fit_paths = []
        cached_sigs = {}
        val_cls = test_cls = None
        val_valid = test_valid = None
        val_labels = test_labels = None
        depth = None
        positives_sorted = None

    # === Identify new images (not in cache) ===
    new_paths = [p for p in requested_paths if _sig(p) not in cached_sigs]
    n_new = len(new_paths)
    n_already = len(requested_paths) - n_new
    logging.info(
        f"requested {args.in_dist_subset}: {len(requested_paths)} images. "
        f"{n_already} already cached, {n_new} NEW to extract."
    )

    need_extractor = (n_new > 0) or (val_cls is None)
    if need_extractor:
        extractor = LocalDinoV2Extractor(
            config_file=args.config_file,
            pretrained_weights=args.pretrained_weights,
            opts=args.opts,
            submodule_path=args.local_submodule_path,
            norm_mean=tuple(args.local_norm_mean),
            norm_std=tuple(args.local_norm_std),
        )
        if depth is None:
            depth = (
                len(extractor.model.blocks[-1]) if extractor.model.chunked_blocks
                else len(extractor.model.blocks)
            )
            positives_sorted = sorted({(li if li >= 0 else depth + li) for li in LAYER_UNION})

        # Extract val + test if cache doesn't have them yet
        if val_cls is None:
            val_paths, val_labels_list = list_test_paths(id_test_root, ood_root, split="val")
            test_paths, test_labels_list = list_test_paths(id_test_root, ood_root, split="test")
            val_labels = np.array(val_labels_list)
            test_labels = np.array(test_labels_list)
            val_cls,  val_valid  = extract_per_layer_cls(extractor, val_paths,  args.batch_size, args.image_res, "val")
            test_cls, test_valid = extract_per_layer_cls(extractor, test_paths, args.batch_size, args.image_res, "test")
            logging.info(
                f"extracted val={val_cls.shape}, test={test_cls.shape}"
            )

        # Extract new fit images
        if n_new > 0:
            new_cls, new_valid = extract_per_layer_cls(extractor, new_paths, args.batch_size, args.image_res, "fit_new")
            new_paths_kept = [p for p, v in zip(new_paths, new_valid) if v]
            n_dropped = n_new - len(new_paths_kept)
            if n_dropped > 0:
                logging.warning(
                    f"{n_dropped} fit image(s) unreadable; "
                    f"the integrity check below will fail"
                )
            logging.info(f"extracted {len(new_paths_kept)} new fit images: {new_cls.shape}")
            if cached_fit_cls is None:
                fit_cls = new_cls
                fit_paths = new_paths_kept
            else:
                fit_cls = np.concatenate([cached_fit_cls, new_cls], axis=0)
                fit_paths = cached_fit_paths + new_paths_kept
        else:
            fit_cls = cached_fit_cls
            fit_paths = cached_fit_paths
    else:
        # Nothing new — keep existing cache contents.
        fit_cls = cached_fit_cls
        fit_paths = cached_fit_paths

    # === Save subset_idx for THIS size; carry forward prior sizes ===
    sig_to_idx_full = {_sig(p): i for i, p in enumerate(fit_paths)}
    this_key = f"subset_idx_{args.size}"
    missing = [p for p in requested_paths if _sig(p) not in sig_to_idx_full]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(requested_paths)} requested images not in cache "
            f"after extraction — fit_paths/fit_cls likely misaligned"
        )
    this_idx = np.array(
        [sig_to_idx_full[_sig(p)] for p in requested_paths], dtype=np.int64
    )
    subset_idx_arrays = dict(prior_subset_idx)
    subset_idx_arrays[this_key] = this_idx
    logging.info(
        f"  {this_key}: {len(this_idx)} indices saved; "
        f"cache now has {sorted(subset_idx_arrays.keys())}"
    )

    if fit_cls is None or len(fit_paths) == 0:
        raise RuntimeError(
            f"No fit images for {args.in_dist_subset}; nothing to save. "
            f"Did you pass an empty subset?"
        )

    # === Save ===
    args.out_cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_cache_file,
        fit_cls=fit_cls,
        fit_paths=np.array(fit_paths),
        val_cls=val_cls,
        test_cls=test_cls,
        val_valid=val_valid,
        test_valid=test_valid,
        val_labels=val_labels,
        test_labels=test_labels,
        depth=np.array(depth, dtype=np.int32),
        positives_sorted=np.array(positives_sorted, dtype=np.int32),
        image_res=np.array(args.image_res, dtype=np.int32),
        **subset_idx_arrays,
    )
    size_gb = args.out_cache_file.stat().st_size / 1e9
    logging.info(
        f"saved {args.out_cache_file} ({size_gb:.2f} GB) — "
        f"fit_cls shape {fit_cls.shape}, {len(subset_idx_arrays)} subset_idx arrays"
    )


if __name__ == "__main__":
    main()
