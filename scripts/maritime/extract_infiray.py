"""Infiray feature extraction → one cache per (backbone, variant).

Builds a maritime-schema .npz the existing grid/eval scripts can consume:
  fit_cls    train (all 7 cats)         [N_train, 24, 1024]
  fit_cats   train category ids          [N_train]
  subset_idx_idonly  indices of ID-category train rows (for `--size all` fit)
  val_cls    root A val (this variant)   [700, 24, 1024]   (Infiray "val")
  test_cls   root B val (this variant)   [700, 24, 1024]   (Infiray "test", _Reversed)
  val/test_labels  5 ID (0) / 2 OOD (1); val/test_valid; depth; positives_sorted; image_res

Same checkpoints/normalization/opts as the maritime extraction; only --pretrained_weights
differs between backbones. One cache per call: e.g.
  --backbone base --variant raw --out_cache_file .../infiray_base_raw.npz
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from subspacead.core.extractor_local import LocalDinoV2Extractor
from subspacead.data import infiray
# reuse the local-backbone forward + layer union from the maritime extractor,
# driven by the shared batched extraction loop
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import local_forward, LAYER_UNION
from _common import extract_per_layer_cls

ROOT_A = "/data/hanchong/other-infrared-datasets/processed-data/红外船舶数据库"          # val source
ROOT_B = "/data/hanchong/other-infrared-datasets/processed-data/红外船舶数据库_Reversed"  # test source
CKPTS = {
    "base":      "/data/hanchong/artifacts-dinov2-all/pretraining/ViT-L-16/eval/training_2348399/teacher_checkpoint.pth",
    "continual": "/data/hanchong/artifacts-dinov2-all/pretraining/ViT-L-16-Continual-IR/eval/training_51199/teacher_checkpoint.pth",
}
CONFIG = "/home/hcchua/Maritime-Vessel-Recognition/submodules/dinov2/dinov2/configs/train/vitl16_short.yaml"
SUBMOD = "/home/hcchua/Maritime-Vessel-Recognition/submodules/dinov2"
OPTS = ["student.arch=vit_large", "student.block_chunks=4", "student.num_register_tokens=4",
        "student.interpolate_antialias=true", "student.interpolate_offset=0.0"]


def main():
    p = argparse.ArgumentParser(description="Infiray feature extraction (one cache per backbone×variant).")
    p.add_argument("--backbone", required=True, choices=["base", "continual"])
    p.add_argument("--variant", required=True, choices=["raw", "enhwo", "enhw"])
    p.add_argument("--out_cache_file", type=Path, required=True)
    p.add_argument("--image_res", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[infiray] %(asctime)s %(message)s")
    ckpt = CKPTS[args.backbone]

    extractor = LocalDinoV2Extractor(
        config_file=CONFIG, pretrained_weights=ckpt, opts=OPTS, submodule_path=SUBMOD,
        norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5),
    )
    depth = (len(extractor.model.blocks[-1]) if extractor.model.chunked_blocks
             else len(extractor.model.blocks))
    positives_sorted = sorted({(li if li >= 0 else depth + li) for li in LAYER_UNION})
    forward_batch = local_forward(extractor, positives_sorted, args.image_res)

    # --- fit: train, all 7 categories (root A) ---
    fit_paths, fit_cats = infiray.list_split(ROOT_A, "train", args.variant)
    fit_cls, fit_valid, *_ = extract_per_layer_cls(fit_paths, args.batch_size, args.image_res, "fit", forward_batch)
    fit_cats = np.array(fit_cats)[fit_valid]
    # indices of ID-category rows, into the (valid-filtered) fit_cls
    idset = set(infiray.ID_CATS)
    subset_idx_idonly = np.array([i for i, c in enumerate(fit_cats) if c in idset], dtype=np.int64)

    # --- val = root A <variant>, test = root B <variant> (both all 7 cats, 5 ID / 2 OOD labels) ---
    val_paths, val_cats = infiray.list_split(ROOT_A, "val", args.variant)
    test_paths, test_cats = infiray.list_split(ROOT_B, "val", args.variant)
    val_cls, val_valid, *_ = extract_per_layer_cls(val_paths, args.batch_size, args.image_res, "val", forward_batch)
    test_cls, test_valid, *_ = extract_per_layer_cls(test_paths, args.batch_size, args.image_res, "test", forward_batch)
    val_labels = np.array(infiray.labels_from_cats(val_cats))
    test_labels = np.array(infiray.labels_from_cats(test_cats))

    args.out_cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_cache_file,
        fit_cls=fit_cls, fit_cats=fit_cats, subset_idx_idonly=subset_idx_idonly,
        val_cls=val_cls, val_valid=val_valid, val_labels=val_labels,
        test_cls=test_cls, test_valid=test_valid, test_labels=test_labels,
        depth=np.array(depth, dtype=np.int32),
        positives_sorted=np.array(positives_sorted, dtype=np.int32),
        image_res=np.array(args.image_res, dtype=np.int32),
    )
    size_gb = args.out_cache_file.stat().st_size / 1e9
    logging.info(
        f"saved {args.out_cache_file} ({size_gb:.2f} GB) — "
        f"fit {fit_cls.shape} (ID rows {len(subset_idx_idonly)}), "
        f"val {val_cls.shape} (ID {int((val_labels==0).sum())} OOD {int((val_labels==1).sum())}), "
        f"test {test_cls.shape} (ID {int((test_labels==0).sum())} OOD {int((test_labels==1).sum())})"
    )


if __name__ == "__main__":
    main()
