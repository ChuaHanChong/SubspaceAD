"""Ask-2 (revised): maritime-fit detectors GRID-evaluated on Infiray, val-selected.

For ONE experiment (a maritime fit cache + size + backbone):
  negatives = Infiray (all 7 cats, treated as in-distribution) — val = root A,
              test = root B (红外船舶数据库_Reversed)
  positives = that experiment's matched maritime OOD rows (val / test)
              (original-IR OOD for original experiments, degraded-IR OOD for the
               degraded ones — same per-experiment matching as the study).

The full paper grid (layers × agg × ev × score × drop_k) is run with PCA fit on
the maritime fit subset (cross-domain). PCA is fit ONCE per (layers, agg, ev) and
scored against all 3 Infiray eval variants (raw / enhwo / enhw). Selection is by
Infiray-val AUROC; the val-derived threshold is applied to test (no leakage).

Writes results_maritime/grids/infcross_{exp}_full.csv (every config, every variant).
Run one experiment per GPU via crossdomain.sh; infcross_report.py builds the Variant A/B reports.
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from _common import (
    AGG_METHODS, DROP_KS, EVS, SCORE_METHODS,
    _chunked, aggregate_features, compute_metrics, make_layer_configs,
)
from subspacead.core.pca import PCAModel
from subspacead.post_process.scoring import calculate_anomaly_scores

CACHE = Path("/data/hanchong/subspacead_cache")
GRIDS = Path("/home/hcchua/SubspaceAD/results_maritime/grids")
VARIANTS = ["raw", "enhwo", "enhw"]

# exp -> (fit_cache, ood_cache, infiray_backbone, fit_size)  — sizes = maritime val-best.
EXP = {
    "ir":        ("ir.npz",                    "ir.npz",                    "base",      10000),
    "irX":       ("ir.npz",                    "ir_degraded.npz",           "base",      10000),
    "irdeg":     ("ir_degraded.npz",           "ir_degraded.npz",           "base",       1000),
    "rgbir":     ("rgbir_pool.npz",            "rgbir_pool.npz",            "base",      10000),
    "rgbirX":    ("rgbir_pool_origRGB_degIR.npz", "rgbir_pool_origRGB_degIR.npz", "base",   5000),
    "ircont":    ("ir_continual.npz",          "ir_continual.npz",          "continual",  5000),
    "irXcont":   ("ir_continual.npz",          "ir_degraded_continual.npz", "continual",  1000),
    "irdegcont": ("ir_degraded_continual.npz", "ir_degraded_continual.npz", "continual",  5000),
}


def _ood_rows(de, split):
    """Maritime OOD (label==1) feature rows for a split, after the valid mask."""
    cls = de[f"{split}_cls"][de[f"{split}_valid"]]
    lab = de[f"{split}_labels"][de[f"{split}_valid"]]
    return cls[lab == 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, choices=list(EXP))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"[infcross:{args.experiment}] %(asctime)s %(message)s")

    fitf, oodf, backbone, size = EXP[args.experiment]

    d = np.load(CACHE / fitf, allow_pickle=False)
    depth = int(d["depth"]); pos = d["positives_sorted"].tolist()
    fit_cls = d["fit_cls"][d[f"subset_idx_{size}"]]      # maritime fit subset
    logging.info(f"fit {fitf} subset_idx_{size}: {fit_cls.shape}  depth={depth}")
    del d

    de = np.load(CACHE / oodf, allow_pickle=False)
    if int(de["depth"]) != depth or de["positives_sorted"].tolist() != pos:
        raise SystemExit(f"fit/ood cache depth/positives mismatch ({oodf})")
    ood_val, ood_test = _ood_rows(de, "val"), _ood_rows(de, "test")
    del de

    inf = {}                                              # variant -> (neg_val rootA, neg_test rootB)
    for v in VARIANTS:
        c = np.load(CACHE / f"infiray_{backbone}_{v}.npz", allow_pickle=False)
        if int(c["depth"]) != depth or c["positives_sorted"].tolist() != pos:
            raise SystemExit(f"infiray_{backbone}_{v} depth/positives mismatch")
        inf[v] = (c["val_cls"][c["val_valid"]], c["test_cls"][c["test_valid"]])
    logging.info(f"OOD pos val={len(ood_val)} test={len(ood_test)} | "
                 f"infiray neg val={inf['raw'][0].shape[0]} test={inf['raw'][1].shape[0]}")

    layer_configs = make_layer_configs(depth)
    n_fit = fit_cls.shape[0]
    rows, n_done = [], 0
    for layer_tag, layers in layer_configs.items():
        for agg in AGG_METHODS:
            if agg == "concat" and n_fit < len(layers) * fit_cls.shape[-1]:
                continue                                  # rank-deficient PCA
            fit_feats   = aggregate_features(fit_cls,  pos, layers, depth, agg)
            ood_val_f   = aggregate_features(ood_val,  pos, layers, depth, agg)
            ood_test_f  = aggregate_features(ood_test, pos, layers, depth, agg)
            neg_f = {v: (aggregate_features(inf[v][0], pos, layers, depth, agg),
                         aggregate_features(inf[v][1], pos, layers, depth, agg)) for v in VARIANTS}

            chunk = 1024
            nfd = fit_feats.shape[0]; nb = (nfd + chunk - 1) // chunk; fd = fit_feats.shape[1]
            for ev in EVS:
                pca = PCAModel(ev=ev).fit(_chunked(fit_feats, chunk), fd, nfd, nb)
                k = int(pca["k"])
                for score in SCORE_METHODS:
                    for dk in DROP_KS:
                        pv = calculate_anomaly_scores(ood_val_f,  pca, score, drop_k=dk)
                        pt = calculate_anomaly_scores(ood_test_f, pca, score, drop_k=dk)
                        for v in VARIANTS:
                            nv = calculate_anomaly_scores(neg_f[v][0], pca, score, drop_k=dk)
                            nt = calculate_anomaly_scores(neg_f[v][1], pca, score, drop_k=dk)
                            yv = np.concatenate([np.zeros(len(nv), int), np.ones(len(pv), int)])
                            yt = np.concatenate([np.zeros(len(nt), int), np.ones(len(pt), int)])
                            vm = compute_metrics(yv, np.concatenate([nv, pv]))
                            tm = compute_metrics(yt, np.concatenate([nt, pt]), threshold=vm["threshold"])
                            rows.append({
                                "experiment": args.experiment, "backbone": backbone, "variant": v,
                                "layer": layer_tag, "agg": agg, "ev": ev,
                                "score": score, "drop_k": dk, "pca_k": k,
                                "val_auroc": round(vm["auroc"], 4), "val_aupr": round(vm["aupr"], 4),
                                "test_auroc": round(tm["auroc"], 4), "test_aupr": round(tm["aupr"], 4),
                                "test_fpr95": round(tm["fpr_at_95tpr"], 4),
                                "n_neg_val": len(nv), "n_pos_val": len(pv),
                                "n_neg_test": len(nt), "n_pos_test": len(pt),
                            })
                        n_done += 1
                        if n_done % 200 == 0:
                            logging.info(f"progress: {n_done} cfg×{len(VARIANTS)}var")

    GRIDS.mkdir(parents=True, exist_ok=True)
    out = GRIDS / f"infcross_{args.experiment}_full.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for v in VARIANTS:
        b = max((r for r in rows if r["variant"] == v), key=lambda r: r["val_auroc"])
        logging.info(f"BEST {v}: {b['layer']} {b['agg']} EV{b['ev']} {b['score']} dk{b['drop_k']} "
                     f"val={b['val_auroc']:.4f} test={b['test_auroc']:.4f}")
    logging.info(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
