"""Class-balanced test metrics for every experiment's val-best config.

The natural test split is 1700 ID + 400 OOD (19% positive), which suppresses AUPR
even when ranking (AUROC) is strong. This script re-scores each experiment's
val-best configuration on a class-balanced subsample (400 ID + 400 OOD), keeping
the val-derived Youden threshold fixed, and reports:

  - Imbalanced baseline (full test)
  - Balanced multi-seed mean ± std across N random subsamples

Configs are read from best_per_experiment.csv (no hardcoding); fit and eval caches
are resolved per experiment so cross-domain runs (irX, irXcont) use the right pair.
Results print to stdout and are written to results_maritime/balanced_test.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from subspacead.core.pca import PCAModel
from subspacead.post_process.scoring import calculate_anomaly_scores
from anomaly_detection import aggregate_features, _chunked, compute_metrics, LAYER_CONFIGS

CACHE = Path("/data/hanchong/subspacead_cache")
RESULTS = Path("/home/hcchua/SubspaceAD/results_maritime")

# experiment (dataset tag in best_per_experiment.csv) -> (fit cache, eval cache)
CACHES = {
    "rgb":       (CACHE / "rgb.npz",                   CACHE / "rgb.npz"),
    "ir":        (CACHE / "ir.npz",                    CACHE / "ir.npz"),
    "irX":       (CACHE / "ir.npz",                    CACHE / "ir_degraded.npz"),
    "irdeg":     (CACHE / "ir_degraded.npz",           CACHE / "ir_degraded.npz"),
    "rgbir":     (CACHE / "rgbir_pool.npz",            CACHE / "rgbir_pool.npz"),
    "ircont":    (CACHE / "ir_continual.npz",          CACHE / "ir_continual.npz"),
    "irXcont":   (CACHE / "ir_continual.npz",          CACHE / "ir_degraded_continual.npz"),
    "irdegcont": (CACHE / "ir_degraded_continual.npz", CACHE / "ir_degraded_continual.npz"),
}


def evaluate(tag, dataset, size, layers, agg, ev, score, drop_k, n_seeds=20):
    fit_cache, eval_cache = CACHES[dataset]
    df = np.load(fit_cache, allow_pickle=False)
    depth = int(df["depth"])
    positives_sorted = df["positives_sorted"].tolist()
    fit_cls = df["fit_cls"][df[f"subset_idx_{size}"]]

    de = df if eval_cache == fit_cache else np.load(eval_cache, allow_pickle=False)
    val_valid, test_valid = de["val_valid"], de["test_valid"]
    val_cls, test_cls = de["val_cls"], de["test_cls"]
    val_labels = de["val_labels"][val_valid]
    test_labels = de["test_labels"][test_valid]

    fit_feats  = aggregate_features(fit_cls,  positives_sorted, layers, depth, agg)
    val_feats  = aggregate_features(val_cls,  positives_sorted, layers, depth, agg)
    test_feats = aggregate_features(test_cls, positives_sorted, layers, depth, agg)

    chunk = 1024
    n = fit_feats.shape[0]
    pca = PCAModel(ev=ev).fit(_chunked(fit_feats, chunk), fit_feats.shape[1], n, (n + chunk - 1) // chunk)
    val_scores  = calculate_anomaly_scores(val_feats,  pca, score, drop_k=drop_k)
    test_scores = calculate_anomaly_scores(test_feats, pca, score, drop_k=drop_k)

    val_m  = compute_metrics(val_labels, val_scores)
    thr = val_m["threshold"]
    imb = compute_metrics(test_labels, test_scores, threshold=thr)

    id_idx = np.where(test_labels == 0)[0]
    ood_idx = np.where(test_labels == 1)[0]
    n_min = min(len(id_idx), len(ood_idx))
    keys = ["auroc", "aupr", "fpr_at_95tpr", "accuracy", "macro_f1",
            "precision_ood", "recall_ood", "f1_ood"]
    acc = {k: [] for k in keys}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(id_idx, size=n_min, replace=False), ood_idx])
        m = compute_metrics(test_labels[idx], test_scores[idx], threshold=thr)
        for k in keys:
            acc[k].append(m[k])
    bal = {k: (float(np.mean(acc[k])), float(np.std(acc[k]))) for k in keys}

    print(f"\n=== {tag}  (size={size} {','.join(map(str,layers))} {agg} EV={ev} {score} dk={drop_k}) ===")
    print(f"  IMBALANCED ({int((test_labels==0).sum())} ID + {int((test_labels==1).sum())} OOD): "
          f"AUROC={imb['auroc']:.4f} AUPR={imb['aupr']:.4f} acc={imb['accuracy']:.4f}")
    print(f"  BALANCED  ({n_min} ID + {n_min} OOD, {n_seeds} seeds): "
          f"AUROC={bal['auroc'][0]:.4f}±{bal['auroc'][1]:.4f} "
          f"AUPR={bal['aupr'][0]:.4f}±{bal['aupr'][1]:.4f} "
          f"acc={bal['accuracy'][0]:.4f}±{bal['accuracy'][1]:.4f}")

    return {
        "experiment": tag, "dataset": dataset, "size": size, "n_balanced": n_min,
        "imb_auroc": round(imb["auroc"], 4), "imb_aupr": round(imb["aupr"], 4),
        "imb_accuracy": round(imb["accuracy"], 4),
        "bal_auroc": round(bal["auroc"][0], 4), "bal_auroc_std": round(bal["auroc"][1], 4),
        "bal_aupr": round(bal["aupr"][0], 4), "bal_aupr_std": round(bal["aupr"][1], 4),
        "bal_accuracy": round(bal["accuracy"][0], 4),
        "bal_macro_f1": round(bal["macro_f1"][0], 4),
        "bal_ood_precision": round(bal["precision_ood"][0], 4),
        "bal_ood_recall": round(bal["recall_ood"][0], 4),
        "bal_ood_f1": round(bal["f1_ood"][0], 4),
    }


def main():
    best = list(csv.DictReader(open(RESULTS / "best_per_experiment.csv")))
    rows = []
    for r in best:
        layers = LAYER_CONFIGS[r["layer"]]
        rows.append(evaluate(
            tag=r["experiment"], dataset=r["dataset"], size=int(r["size"]),
            layers=layers, agg=r["agg"], ev=float(r["ev"]),
            score=r["score"], drop_k=int(r["drop_k"]),
        ))

    cols = list(rows[0].keys())
    out = RESULTS / "balanced_test.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
