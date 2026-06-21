"""PCA + scoring hyperparameter grid on a cached DINOv2 feature .npz.

Fits the PCA on all of the cache's fit features and sweeps the axes
(layers × agg × EV × score × drop_k), writing one metrics.json per config; aggregate.py
then reports the val-best. Concat configs with N_fit < D are skipped.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from subspacead.core.pca import PCAModel
from subspacead.post_process.scoring import calculate_anomaly_scores

from _common import (
    make_layer_configs, AGG_METHODS, AGG_SHORT, EVS,
    SCORE_METHODS, SCORE_SHORT, DROP_KS, compute_metrics, aggregate_features,
    _chunked,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_cache_file", type=Path, required=True,
                   help=".npz from extract_features.py; supplies the FIT features (fit_cls).")
    p.add_argument("--eval_cache_file", type=Path, default=None,
                   help="optional .npz supplying val/test features + labels; defaults to "
                        "--feature_cache_file. Set it to fit on one dataset and evaluate on "
                        "another (cross-domain). Must share depth/positives_sorted.")
    p.add_argument("--out_prefix", required=True,
                   help="e.g. 'grid_example_' — written to results_root/{prefix}{tag}/")
    p.add_argument("--results_root", type=Path, default=Path("results/grids"))
    args = p.parse_args()

    if not args.feature_cache_file.exists():
        raise FileNotFoundError(f"cache not found: {args.feature_cache_file}")

    # Fit features from --feature_cache_file; eval (val/test) from --eval_cache_file
    # (defaults to the same file). This split allows cross-domain (fit one, eval another).
    d = np.load(args.feature_cache_file, allow_pickle=False)
    fit_cls = d["fit_cls"]
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
            raise RuntimeError("fit and eval caches disagree on depth/positives_sorted")
        print(f"eval features from separate cache: {eval_cache_file.name}")

    val_cls, test_cls = de["val_cls"], de["test_cls"]
    val_labels_v = de["val_labels"][de["val_valid"]]
    test_labels_v = de["test_labels"][de["test_valid"]]

    print(f"fit {fit_cls.shape}; val {val_cls.shape}, test {test_cls.shape}")

    layer_configs = make_layer_configs(depth)   # depth-aware (12-layer ViT-S/B vs 24-layer ViT-L)
    n_total = (len(layer_configs) * len(AGG_METHODS) * len(EVS)
               * len(SCORE_METHODS) * len(DROP_KS))
    n_done = 0
    n_fit = fit_cls.shape[0]

    for layer_tag, layers in layer_configs.items():
        for agg in AGG_METHODS:
            # Skip concat configs where N_fit < D (PCA would be rank-deficient).
            if agg == "concat" and n_fit < len(layers) * fit_cls.shape[-1]:
                print(f"skip {layer_tag} concat: N_fit={n_fit} < D={len(layers)*fit_cls.shape[-1]}")
                continue
            fit_feats  = aggregate_features(fit_cls,  positives_sorted, layers, depth, agg)
            val_feats  = aggregate_features(val_cls,  positives_sorted, layers, depth, agg)
            test_feats = aggregate_features(test_cls, positives_sorted, layers, depth, agg)

            chunk = 1024
            n_fit_d = fit_feats.shape[0]
            n_batches = (n_fit_d + chunk - 1) // chunk
            feat_dim = fit_feats.shape[1]

            for ev in EVS:
                pca_params = PCAModel(ev=ev).fit(_chunked(fit_feats, chunk), feat_dim, n_fit_d, n_batches)
                k = int(pca_params["k"])
                for score_method in SCORE_METHODS:
                    for drop_k in DROP_KS:
                        val_scores  = calculate_anomaly_scores(val_feats,  pca_params, score_method, drop_k=drop_k)
                        test_scores = calculate_anomaly_scores(test_feats, pca_params, score_method, drop_k=drop_k)
                        val_metrics  = compute_metrics(val_labels_v, val_scores)
                        # Apply the val-derived threshold to test (no leakage).
                        test_metrics = compute_metrics(test_labels_v, test_scores,
                                                       threshold=val_metrics["threshold"])
                        tag = (f"{layer_tag}_{AGG_SHORT[agg]}_EV{int(round(ev*1000)):04d}_"
                               f"{SCORE_SHORT[score_method]}_dk{drop_k}")
                        out = {
                            "val": val_metrics, "test": test_metrics, "pca_k": k,
                            "config": {
                                "layers": layers, "layer_tag": layer_tag,
                                "agg_method": agg, "ev": ev,
                                "score_method": score_method, "drop_k": drop_k,
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
                            print(f"progress: {n_done}/{n_total}")

    print(f"done — {n_done}/{n_total} configs written")


if __name__ == "__main__":
    main()
