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
import torch
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)

from subspacead.core.pca import PCAModel
from subspacead.data.maritime import list_fit_paths, resolve_roots
from subspacead.post_process.scoring import calculate_anomaly_scores

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paper-only axes
LAYER_CONFIGS: dict[str, list[int]] = {
    "L1":   [-1],
    "L2":   [-1, -2],
    "L4":   [-1, -2, -3, -4],
    "L6":   [-1, -2, -3, -4, -5, -6],
    "L8":   [-1, -2, -3, -4, -5, -6, -7, -8],
    "L12":  [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12],
    "L18":  [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12,
            -13, -14, -15, -16, -17, -18],
    "Lmid": [-12, -13, -14, -15, -16, -17, -18],   # paper default
}
AGG_METHODS = ["mean", "concat"]
AGG_SHORT = {"mean": "m", "concat": "c"}
EVS = [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]
SCORE_METHODS = ["reconstruction", "mahalanobis", "cosine", "euclidean"]
SCORE_SHORT = {"reconstruction": "rec", "mahalanobis": "mah",
               "cosine": "cos", "euclidean": "euc"}
DROP_KS = [0, 5, 20, 50, 100]


def _fpr_at_tpr(y_true, y_score, target):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, target, side="left")
    return float(fpr[idx]) if idx < len(fpr) else float("nan")


def _youden_threshold(y_true, y_score):
    """Threshold maximizing Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(thresholds[np.argmax(tpr - fpr)])


def compute_metrics(y_true, y_score, threshold=None):
    """Threshold-free (AUROC/AUPR/FPR@95) + classification report at threshold.

    If threshold is None, derive via Youden's J on this split. Caller should
    pass the VAL-derived threshold when scoring TEST to avoid leakage.
    """
    if threshold is None:
        threshold = _youden_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    rep = classification_report(
        y_true, y_pred, target_names=["id", "ood"],
        output_dict=True, zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "aupr": float(average_precision_score(y_true, y_score)),
        "fpr_at_95tpr": _fpr_at_tpr(y_true, y_score, 0.95),
        "threshold": threshold,
        "accuracy": float(rep["accuracy"]),
        "precision_id":  float(rep["id"]["precision"]),
        "recall_id":     float(rep["id"]["recall"]),
        "f1_id":         float(rep["id"]["f1-score"]),
        "precision_ood": float(rep["ood"]["precision"]),
        "recall_ood":    float(rep["ood"]["recall"]),
        "f1_ood":        float(rep["ood"]["f1-score"]),
        "macro_f1":      float(rep["macro avg"]["f1-score"]),
        "weighted_f1":   float(rep["weighted avg"]["f1-score"]),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "score_stats": {
            "n_in_dist": int((y_true == 0).sum()),
            "n_ood": int((y_true == 1).sum()),
            "in_dist_score_mean": float(np.mean(y_score[y_true == 0])),
            "ood_score_mean": float(np.mean(y_score[y_true == 1])),
            "in_dist_score_std": float(np.std(y_score[y_true == 0])),
            "ood_score_std": float(np.std(y_score[y_true == 1])),
        },
    }


def aggregate_features(cls_features, positives_sorted, target_layers, depth, agg_method):
    target_pos = [li if li >= 0 else depth + li for li in target_layers]
    axis_indices = [positives_sorted.index(p) for p in target_pos]
    selected = cls_features[:, axis_indices, :]  # [N, n_target, D]
    if agg_method == "mean":
        return selected.mean(axis=1)
    if agg_method == "concat":
        N, n_target, D = selected.shape
        return selected.reshape(N, n_target * D)
    raise ValueError(f"unknown agg_method: {agg_method}")


def _chunked(arr, chunk: int = 1024):
    """Wrap a numpy array as a (re-callable) generator of chunks for PCAModel.fit."""
    def gen():
        for i in range(0, len(arr), chunk):
            yield arr[i : i + chunk]
    return gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_cache_file", type=Path, required=True,
                   help=".npz produced by extract_features.py; supplies the FIT "
                        "features (fit_cls + subset_idx_{--size}). Must contain "
                        "subset_idx_{--size} (or fall back to path-matching).")
    p.add_argument("--eval_cache_file", type=Path, default=None,
                   help="optional .npz supplying val/test features + labels. "
                        "Defaults to --feature_cache_file. Set this to fit on one "
                        "dataset and evaluate on another (e.g. fit clean IR, "
                        "eval degraded IR). Must share depth/positives_sorted.")
    p.add_argument("--data_root", required=True,
                   help="dataset root; used to re-glob the subset paths for slicing")
    p.add_argument("--in_dist_subset", required=True,
                   help="subset to use for fit (e.g. In-distribution_5000perCat). "
                        "Its train paths must be a subset of those in the cache.")
    p.add_argument("--size", type=int, required=True,
                   help="fit-subset size per class (e.g. 5000). Used to look up "
                        "subset_idx_{size} in the cache.")
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
    fit_paths_full = d["fit_paths"].astype(str)   # [N_max_fit]
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
    idx_key = f"subset_idx_{args.size}"

    if idx_key in d.files:
        fit_idx = d[idx_key]
        logging.info(f"using precomputed {idx_key}: {len(fit_idx)} indices")
    else:
        logging.info(f"{idx_key} not in cache; falling back to path-matching")
        in_dist_root, _ = resolve_roots(Path(args.data_root), args.in_dist_subset, args.ood_subset)
        subset_paths = list_fit_paths(in_dist_root, seed=42)
        def _sig(p):
            pp = Path(p)
            return (pp.parent.name, pp.name)
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

    n_total = (len(LAYER_CONFIGS) * len(AGG_METHODS) * len(EVS)
               * len(SCORE_METHODS) * len(DROP_KS))
    n_done = 0

    n_fit = fit_cls.shape[0]
    for layer_tag, layers in LAYER_CONFIGS.items():
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
