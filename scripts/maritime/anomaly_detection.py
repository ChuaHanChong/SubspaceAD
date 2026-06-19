"""Anomaly detection hyperparameter grid on cached DINOv2 features.

Loads a .npz produced by extract_features.py and runs the paper's
PCA + scoring grid in-memory. Writes one metrics.json per config.

Paper-only hyperparameter axes:
  layers, agg_method, pca_ev, score_method, drop_k.

Total configs per (dataset, size): 8 * 2 * 6 * 4 * 5 = 1920.
Some concat configs may be skipped when N_fit < D (rank-deficient PCA).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from subspacead.core.pca import PCAModel
from subspacead.data.maritime import list_fit_paths, resolve_roots
from subspacead.post_process.scoring import calculate_anomaly_scores

from _common import (
    make_layer_configs, AGG_METHODS, AGG_SHORT, EVS,
    SCORE_METHODS, SCORE_SHORT, DROP_KS, compute_metrics, aggregate_features,
    _chunked, _sig,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_cache_file", type=Path, required=True,
                   help=".npz produced by extract_features.py; supplies the FIT "
                        "features (fit_cls + subset_idx_{--size}). Must contain "
                        "subset_idx_{--size} (or fall back to path-matching).")
    p.add_argument("--eval_cache_file", type=Path, default=None,
                   help="optional .npz supplying val/test features + labels. "
                        "Defaults to --feature_cache_file. Set this to fit on one "
                        "dataset and evaluate on another (e.g. fit original IR, "
                        "eval degraded IR). Must share depth/positives_sorted.")
    p.add_argument("--data_root", required=True,
                   help="dataset root; used to re-glob the subset paths for slicing")
    p.add_argument("--in_dist_subset", required=True,
                   help="subset to use for fit (e.g. In-distribution_5000perCat). "
                        "Its train paths must be a subset of those in the cache.")
    p.add_argument("--size", required=True,
                   help="fit-subset size per class (e.g. 5000) → subset_idx_{size}; "
                        "or 'all' → subset_idx_idonly (or every fit row).")
    p.add_argument("--ood_subset", default="Out-of-distribution")
    p.add_argument("--out_prefix", required=True,
                   help="e.g. 'grid_rgb_5000_' — written to results_root/{prefix}{tag}/")
    p.add_argument("--results_root", type=Path,
                   default=Path("/home/hcchua/SubspaceAD/results_maritime"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format=f"[{args.out_prefix}] %(asctime)s %(message)s")

    if not args.feature_cache_file.exists():
        raise FileNotFoundError(f"cache not found: {args.feature_cache_file}")

    # Fit features come from --feature_cache_file; eval (val/test) features come
    # from --eval_cache_file (defaults to the same file). This separation lets us
    # fit on one dataset and evaluate on another (cross-domain).
    d = np.load(args.feature_cache_file, allow_pickle=False)
    fit_cls_full   = d["fit_cls"]                 # [N_max_fit, 24, D]
    fit_paths_full = d["fit_paths"].astype(str) if "fit_paths" in d.files else None  # path-match fallback only
    depth = int(d["depth"])
    positives_sorted = d["positives_sorted"].tolist()

    eval_cache_file = args.eval_cache_file or args.feature_cache_file
    if eval_cache_file == args.feature_cache_file:
        de = d
    else:
        if not eval_cache_file.exists():
            raise FileNotFoundError(f"eval cache not found: {eval_cache_file}")
        de = np.load(eval_cache_file, allow_pickle=False)
        if int(de["depth"]) != depth or de["positives_sorted"].tolist() != positives_sorted:
            raise RuntimeError(
                "fit and eval caches disagree on depth/positives_sorted; "
                "they must come from the same extraction config"
            )
        logging.info(f"eval features from separate cache: {eval_cache_file.name}")

    val_cls   = de["val_cls"]
    test_cls  = de["test_cls"]
    val_valid  = de["val_valid"]
    test_valid = de["test_valid"]
    val_labels  = de["val_labels"]
    test_labels = de["test_labels"]
    val_labels_v  = val_labels[val_valid]
    test_labels_v = test_labels[test_valid]

    # Prefer the precomputed subset_idx_{size} array in the cache; fall back to
    # path-matching by (class_dir, basename) if not present.
    # --size all → fit on subset_idx_idonly (e.g. Infiray ID categories) or every fit row.
    if args.size == "all":
        fit_idx = (d["subset_idx_idonly"] if "subset_idx_idonly" in d.files
                   else np.arange(fit_cls_full.shape[0], dtype=np.int64))
        logging.info(f"--size all: fitting on {len(fit_idx)} rows")
    elif (idx_key := f"subset_idx_{args.size}") in d.files:
        fit_idx = d[idx_key]
        logging.info(f"using precomputed {idx_key}: {len(fit_idx)} indices")
    else:
        logging.info(f"{idx_key} not in cache; falling back to path-matching")
        if fit_paths_full is None:
            raise RuntimeError(
                f"cache has neither subset_idx_{args.size} nor fit_paths; "
                f"pass --size all or a size present in the cache"
            )
        in_dist_root, _ = resolve_roots(Path(args.data_root), args.in_dist_subset, args.ood_subset)
        subset_paths = list_fit_paths(in_dist_root, seed=42)
        sig_to_idx = {_sig(p): i for i, p in enumerate(fit_paths_full)}
        missing = [p for p in subset_paths if _sig(p) not in sig_to_idx]
        if missing:
            raise RuntimeError(
                f"{len(missing)} subset image(s) not found in cache "
                f"(cache must cover the largest size); first 3 missing basenames: "
                f"{[Path(p).name for p in missing[:3]]}"
            )
        fit_idx = np.array([sig_to_idx[_sig(p)] for p in subset_paths])
    fit_cls = fit_cls_full[fit_idx]
    logging.info(
        f"cache {args.feature_cache_file.name}: full fit={fit_cls_full.shape[0]}; "
        f"using subset of {fit_cls.shape[0]} for {args.in_dist_subset}. "
        f"val={val_cls.shape}, test={test_cls.shape}"
    )

    layer_configs = make_layer_configs(depth)   # depth-aware (12-layer ViT-S/B vs 24-layer ViT-L)
    logging.info(f"depth={depth} → layer configs: {list(layer_configs)}")
    n_total = (len(layer_configs) * len(AGG_METHODS) * len(EVS)
               * len(SCORE_METHODS) * len(DROP_KS))
    n_done = 0

    n_fit = fit_cls.shape[0]
    for layer_tag, layers in layer_configs.items():
        for agg in AGG_METHODS:
            # Skip concat configs where N_fit < D (PCA would be rank-deficient).
            if agg == "concat":
                d_concat = len(layers) * fit_cls.shape[-1]
                if n_fit < d_concat:
                    skipped = len(EVS) * len(SCORE_METHODS) * len(DROP_KS)
                    logging.info(
                        f"skip {layer_tag} concat: N_fit={n_fit} < D={d_concat} "
                        f"(would be rank-deficient); {skipped} configs not run"
                    )
                    continue
            fit_feats  = aggregate_features(fit_cls,  positives_sorted, layers, depth, agg)
            val_feats  = aggregate_features(val_cls,  positives_sorted, layers, depth, agg)
            test_feats = aggregate_features(test_cls, positives_sorted, layers, depth, agg)

            chunk = 1024
            n_fit_d = fit_feats.shape[0]
            n_batches = (n_fit_d + chunk - 1) // chunk
            feat_dim = fit_feats.shape[1]

            for ev in EVS:
                pca_params = PCAModel(ev=ev).fit(
                    _chunked(fit_feats, chunk), feat_dim, n_fit_d, n_batches
                )
                k = int(pca_params["k"])

                for score_method in SCORE_METHODS:
                    for drop_k in DROP_KS:
                        val_scores  = calculate_anomaly_scores(val_feats,  pca_params, score_method, drop_k=drop_k)
                        test_scores = calculate_anomaly_scores(test_feats, pca_params, score_method, drop_k=drop_k)

                        val_metrics  = compute_metrics(val_labels_v,  val_scores)
                        # Apply val-derived threshold to test to avoid leakage.
                        test_metrics = compute_metrics(
                            test_labels_v, test_scores,
                            threshold=val_metrics["threshold"],
                        )

                        tag = (
                            f"{layer_tag}_{AGG_SHORT[agg]}_"
                            f"EV{int(round(ev*1000)):04d}_"
                            f"{SCORE_SHORT[score_method]}_dk{drop_k}"
                        )
                        out = {
                            "val":  val_metrics,
                            "test": test_metrics,
                            "pca_k": k,
                            "config": {
                                "layers": layers,
                                "layer_tag": layer_tag,
                                "agg_method": agg,
                                "ev": ev,
                                "score_method": score_method,
                                "drop_k": drop_k,
                                "feature_cache_file": str(args.feature_cache_file),
                                "eval_cache_file": str(eval_cache_file),
                                "token_type": "cls",
                            },
                        }
                        outdir = args.results_root / f"{args.out_prefix}{tag}"
                        outdir.mkdir(parents=True, exist_ok=True)
                        (outdir / "metrics.json").write_text(json.dumps(out, indent=2))
                        n_done += 1
                        if n_done % 200 == 0 or n_done == n_total:
                            logging.info(f"progress: {n_done}/{n_total}")

    logging.info(f"done — {n_done}/{n_total} configs written")


if __name__ == "__main__":
    main()
