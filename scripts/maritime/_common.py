"""Shared, dependency-light helpers for extract_features.py and anomaly_detection.py.

Provides the hyperparameter axes, depth-aware layer configs, metric helpers, feature
aggregation, the chunked-fit generator, a recursive image lister, and a model-agnostic
CLS-extraction loop (the model forward is injected as a `forward_batch` callable).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Layer configs are depth-aware: a 24-layer backbone (ViT-L) gets the full set
# incl. L18 + Lmid[-12..-18]; a 12-layer backbone (ViT-S/B) drops the out-of-range
# configs and uses a middle-band Lmid.
def make_layer_configs(depth: int) -> dict:
    base = {
        "L1":  [-1],
        "L2":  [-1, -2],
        "L4":  list(range(-1, -5, -1)),
        "L6":  list(range(-1, -7, -1)),
        "L8":  list(range(-1, -9, -1)),
        "L12": list(range(-1, -13, -1)),
        "L18": list(range(-1, -19, -1)),
    }
    cfg = {k: v for k, v in base.items() if max(-x for x in v) <= depth}
    if depth >= 18:
        cfg["Lmid"] = list(range(-12, -19, -1))          # default (24-layer)
    else:
        mid = depth // 2                                  # 12-layer → [-5,-6,-7,-8]
        cfg["Lmid"] = [-(mid - 1), -mid, -(mid + 1), -(mid + 2)]
    return cfg


AGG_METHODS = ["mean", "concat"]
AGG_SHORT = {"mean": "m", "concat": "c"}
EVS = [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]
SCORE_METHODS = ["reconstruction", "mahalanobis", "cosine", "euclidean"]
SCORE_SHORT = {"reconstruction": "rec", "mahalanobis": "mah",
               "cosine": "cos", "euclidean": "euc"}
DROP_KS = [0, 5, 20, 50, 100]


def _youden_threshold(y_true, y_score):
    """Threshold maximizing Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(thresholds[np.argmax(tpr - fpr)])


def compute_metrics(y_true, y_score, threshold=None):
    """Threshold-free (AUROC/AUPR) + classification report at threshold.

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
            "n_id": int((y_true == 0).sum()),
            "n_ood": int((y_true == 1).sum()),
            "id_score_mean": float(np.mean(y_score[y_true == 0])),
            "ood_score_mean": float(np.mean(y_score[y_true == 1])),
            "id_score_std": float(np.std(y_score[y_true == 0])),
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


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder):
    """Recursively list image files under `folder`, sorted (deterministic order)."""
    root = Path(folder)
    return sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


@torch.no_grad()
def extract_per_layer_cls(paths, batch_size, desc, forward_batch):
    """Model-agnostic batched per-layer CLS extraction.

    Opens each path as an RGB image; unreadable ones are reported and skipped
    (valid[i] stays False). `forward_batch(pil_list)` must return a Tensor of
    shape [B, depth, D] (the CLS token per layer).

    Returns (cls_all [N, depth, D] float32, valid bool[len(paths)]).
    """
    cls_chunks, valid = [], np.zeros(len(paths), dtype=bool)
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        pil_imgs, local_idx = [], []
        for k, p in enumerate(paths[i : i + batch_size]):
            try:
                pil_imgs.append(Image.open(p).convert("RGB"))
                local_idx.append(k)
            except Exception as e:
                print(f"skip unreadable {p}: {e}")
        if not pil_imgs:
            continue
        cls_chunks.append(forward_batch(pil_imgs).cpu())
        for k in local_idx:
            valid[i + k] = True
    cls_all = torch.cat(cls_chunks, dim=0).numpy().astype(np.float32)
    return cls_all, valid
