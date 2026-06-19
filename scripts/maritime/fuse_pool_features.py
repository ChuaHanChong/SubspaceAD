"""Pool two single-modality feature caches into one (Exp 3: RGB ∪ IR).

"Treat each RGB and each IR image as a separate sample": stack the two caches
along the SAMPLE axis (axis 0), NOT along the feature axis. Feature dim and
layer count are unchanged; the pooled dataset simply has 2× the samples. PCA is
then fit on the union, and every RGB and IR image is scored independently.

The output .npz has the exact same schema as a single-dataset cache, so
anomaly_detection.py / the grid stage consume it with no changes.

  fit_cls          = [rgb_fit ; ir_fit]              # [2N, 24, D]
  val/test_cls     = [rgb_eval ; ir_eval]            # rows pooled, labels too
  subset_idx_{n}   = [rgb_idx ; ir_idx + len(rgb_fit)]   # indexes the stacked fit array
  fit_paths        = modality-prefixed to avoid basename collisions

Pure numpy merge — no DINOv2 forwards.
"""

import argparse
from pathlib import Path

import numpy as np

CACHE_DIR = Path("/data/hanchong/subspacead_cache")


def _check_compatible(a, b):
    """fit/eval caches must share extraction geometry to be poolable."""
    if int(a["depth"]) != int(b["depth"]):
        raise RuntimeError(f"depth mismatch: {int(a['depth'])} vs {int(b['depth'])}")
    if a["positives_sorted"].tolist() != b["positives_sorted"].tolist():
        raise RuntimeError("positives_sorted mismatch between caches")
    if int(a["image_res"]) != int(b["image_res"]):
        raise RuntimeError(f"image_res mismatch: {int(a['image_res'])} vs {int(b['image_res'])}")
    if a["fit_cls"].shape[1:] != b["fit_cls"].shape[1:]:
        raise RuntimeError(f"feature shape mismatch: {a['fit_cls'].shape} vs {b['fit_cls'].shape}")


def main():
    p = argparse.ArgumentParser(description="Pool two modality caches along the sample axis.")
    p.add_argument("--cache_a", type=Path, default=CACHE_DIR / "rgb.npz")
    p.add_argument("--cache_b", type=Path, default=CACHE_DIR / "ir.npz")
    p.add_argument("--tag_a", default="rgb", help="modality prefix for cache_a paths")
    p.add_argument("--tag_b", default="ir",  help="modality prefix for cache_b paths")
    p.add_argument("--out_cache_file", type=Path, default=CACHE_DIR / "rgbir_pool.npz")
    args = p.parse_args()

    a = np.load(args.cache_a, allow_pickle=False)
    b = np.load(args.cache_b, allow_pickle=False)
    _check_compatible(a, b)

    n_a_fit = a["fit_cls"].shape[0]

    # --- fit: stack samples; prefix paths to avoid basename collisions ---
    fit_cls = np.concatenate([a["fit_cls"], b["fit_cls"]], axis=0)
    fit_paths = np.array(
        [f"{args.tag_a}/{q}" for q in a["fit_paths"].astype(str)]
        + [f"{args.tag_b}/{q}" for q in b["fit_paths"].astype(str)]
    )

    # --- eval: pool rows + labels for val and test ---
    val_cls    = np.concatenate([a["val_cls"],   b["val_cls"]],   axis=0)
    test_cls   = np.concatenate([a["test_cls"],  b["test_cls"]],  axis=0)
    val_valid  = np.concatenate([a["val_valid"],  b["val_valid"]])
    test_valid = np.concatenate([a["test_valid"], b["test_valid"]])
    val_labels  = np.concatenate([a["val_labels"],  b["val_labels"]])
    test_labels = np.concatenate([a["test_labels"], b["test_labels"]])

    # --- subset indices: b's indices shift by len(a_fit) into the stacked array ---
    subset_idx = {}
    a_keys = {k for k in a.files if k.startswith("subset_idx_")}
    b_keys = {k for k in b.files if k.startswith("subset_idx_")}
    for k in sorted(a_keys & b_keys):
        subset_idx[k] = np.concatenate([a[k], b[k] + n_a_fit]).astype(np.int64)
    if a_keys ^ b_keys:
        print(f"  note: subset keys not in both caches, skipped: {sorted(a_keys ^ b_keys)}")

    args.out_cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_cache_file,
        fit_cls=fit_cls,
        fit_paths=fit_paths,
        val_cls=val_cls,
        test_cls=test_cls,
        val_valid=val_valid,
        test_valid=test_valid,
        val_labels=val_labels,
        test_labels=test_labels,
        depth=a["depth"],
        positives_sorted=a["positives_sorted"],
        image_res=a["image_res"],
        **subset_idx,
    )
    size_gb = args.out_cache_file.stat().st_size / 1e9
    print(
        f"wrote {args.out_cache_file} ({size_gb:.1f} GB)\n"
        f"  fit_cls {fit_cls.shape} (= {n_a_fit} {args.tag_a} + {b['fit_cls'].shape[0]} {args.tag_b})\n"
        f"  val_cls {val_cls.shape}  (ID={int((val_labels==0).sum())} OOD={int((val_labels==1).sum())})\n"
        f"  test_cls {test_cls.shape} (ID={int((test_labels==0).sum())} OOD={int((test_labels==1).sum())})\n"
        f"  subset_idx: " + ", ".join(f"{k.split('_')[-1]}={len(v)}" for k, v in subset_idx.items())
    )


if __name__ == "__main__":
    main()
