"""Off-the-shelf HF DINOv2 (ViT-S/B/L) CLS feature extraction on the degraded IR set.

Uses the ORIGINAL SubspaceAD HF path: `transformers` AutoModel + AutoImageProcessor
(ImageNet normalization, patch-14), `token_type=cls` per layer. Produces a maritime-schema
cache (fit_cls / val_cls / test_cls + subset_idx_{size}) so the SAME `anomaly_detection.py`
grid (identical PCAModel fit + paper hyperparameter axes) runs on it unchanged. Records
extraction throughput for the speed comparison.

One cache per model, e.g.:
  --model_ckpt facebook/dinov2-small --out_cache_file .../hf_s_degraded.npz
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from subspacead.data.maritime import list_fit_paths, list_test_paths, resolve_roots

import _common
from _common import _sig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_SIZES = [100, 500, 1000, 5000, 10000]


def hf_forward(model, processor, image_res):
    """Build a forward_batch(pils) -> Tensor[B, depth, D] for an HF DINOv2 model.

    depth = number of transformer layers (hidden_states minus the embedding output);
    CLS token = hidden_states[i][:, 0, :] for each layer i = 1..depth.
    """
    @torch.no_grad()
    def forward_batch(pil):
        inputs = processor(images=pil, return_tensors="pt", do_resize=True,
                           size={"height": image_res, "width": image_res},
                           do_center_crop=False).to(DEVICE)
        out = model(**inputs, output_hidden_states=True, output_attentions=False)
        hs = out.hidden_states                       # tuple len depth+1 (embeddings + layers)
        return torch.stack([h[:, 0, :] for h in hs[1:]], dim=1)  # [B, depth, D]
    return forward_batch


def extract_per_layer_cls_hf(model, processor, paths, batch_size, image_res, desc):
    """Return (cls_all [N, depth, D] float32, valid bool[N], seconds, n_imgs)."""
    return _common.extract_per_layer_cls(
        paths, batch_size, image_res, desc, hf_forward(model, processor, image_res)
    )


def main():
    p = argparse.ArgumentParser(description="HF DINOv2 CLS extraction (ViT-S/B/L) on degraded IR.")
    p.add_argument("--model_ckpt", required=True, help="e.g. facebook/dinov2-small")
    p.add_argument("--data_root", required=True)
    p.add_argument("--id_test_subset", default="In-distribution")
    p.add_argument("--ood_subset", default="Out-of-distribution")
    p.add_argument("--fit_subset", default="In-distribution_10000perCat",
                   help="superset to extract; subset_idx for smaller sizes derived by filename")
    p.add_argument("--out_cache_file", type=Path, required=True)
    p.add_argument("--image_res", type=int, default=224)   # /14 = 16 patches
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[hf-extract] %(asctime)s %(message)s")

    processor = AutoImageProcessor.from_pretrained(args.model_ckpt)
    model = AutoModel.from_pretrained(args.model_ckpt).eval().to(DEVICE)
    ps = model.config.patch_size
    if args.image_res % ps != 0:
        raise ValueError(f"image_res {args.image_res} not divisible by patch_size {ps}")
    logging.info(f"{args.model_ckpt}: hidden_size={model.config.hidden_size} "
                 f"layers={model.config.num_hidden_layers} patch={ps}")

    in_dist_root, ood_root = resolve_roots(Path(args.data_root), args.fit_subset, args.ood_subset)
    id_test_root = Path(args.data_root) / args.id_test_subset
    fit_paths = list_fit_paths(in_dist_root, seed=42)
    val_paths, val_labels = list_test_paths(id_test_root, ood_root, split="val")
    test_paths, test_labels = list_test_paths(id_test_root, ood_root, split="test")

    fit_cls, fit_valid, fit_secs, fit_n = extract_per_layer_cls_hf(
        model, processor, fit_paths, args.batch_size, args.image_res, "fit")
    fit_paths = [p for p, v in zip(fit_paths, fit_valid) if v]
    val_cls, val_valid, *_ = extract_per_layer_cls_hf(model, processor, val_paths, args.batch_size, args.image_res, "val")
    test_cls, test_valid, *_ = extract_per_layer_cls_hf(model, processor, test_paths, args.batch_size, args.image_res, "test")

    depth = fit_cls.shape[1]
    positives_sorted = list(range(depth))
    throughput = fit_n / fit_secs if fit_secs else 0.0
    logging.info(f"extraction throughput: {throughput:.1f} img/s ({fit_n} imgs / {fit_secs:.1f}s), "
                 f"{1000*fit_secs/max(fit_n,1):.2f} ms/img")

    # subset_idx_{size} from the 10000 superset by (class_dir, basename)
    sig_to_idx = {_sig(p): i for i, p in enumerate(fit_paths)}
    subset_idx = {}
    for s in ALL_SIZES:
        sub_root = Path(args.data_root) / f"In-distribution_{s}perCat"
        if not sub_root.exists():
            continue
        sub_paths = list_fit_paths(sub_root, seed=42)
        miss = [q for q in sub_paths if _sig(q) not in sig_to_idx]
        if miss:
            logging.info(f"  size {s}: {len(miss)} not in superset; skip"); continue
        subset_idx[f"subset_idx_{s}"] = np.array([sig_to_idx[_sig(q)] for q in sub_paths], dtype=np.int64)

    args.out_cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_cache_file,
        fit_cls=fit_cls, fit_paths=np.array(fit_paths),
        val_cls=val_cls, val_valid=val_valid, val_labels=np.array(val_labels),
        test_cls=test_cls, test_valid=test_valid, test_labels=np.array(test_labels),
        depth=np.array(depth, dtype=np.int32),
        positives_sorted=np.array(positives_sorted, dtype=np.int32),
        image_res=np.array(args.image_res, dtype=np.int32),
        extract_img_per_s=np.array(throughput, dtype=np.float32),
        extract_ms_per_img=np.array(1000 * fit_secs / max(fit_n, 1), dtype=np.float32),
        **subset_idx,
    )
    logging.info(f"saved {args.out_cache_file} — fit {fit_cls.shape} depth={depth}, "
                 f"val {val_cls.shape}, test {test_cls.shape}, sizes {sorted(subset_idx)}")


if __name__ == "__main__":
    main()
