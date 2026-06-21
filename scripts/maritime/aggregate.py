"""Aggregate grid result dirs — the val-best config per group.

For each dir matching --pattern, reads metrics.json and keeps the config with the highest
validation AUROC per GROUP (= dir name minus the trailing _<layer>_<agg>_EV..._<score>_dk tag),
then writes a compact CSV. This is the hyperparameter-tuning selection step.

  python aggregate.py --pattern 'grid_example_*' --csv results/tables/example_best.csv
"""

import argparse
import csv
import fnmatch
import json
import re
from pathlib import Path

ROOT = Path("results/grids")

# Trailing config tag: _<layer>_<m|c>_EV####_<score>_dk#
TAG = re.compile(r"_(?:L\d+|Lmid)_[mc]_EV\d{4}_(?:rec|mah|cos|euc)_dk\d+$")


def group_of(name: str) -> str:
    return TAG.sub("", name)


def pick_best_rows(results_root: Path, pattern: str):
    best = {}  # group -> (val_auroc, row)
    n = 0
    for d in sorted(results_root.iterdir()):
        if not d.is_dir() or not fnmatch.fnmatch(d.name, pattern):
            continue
        mj = d / "metrics.json"
        if not mj.exists():
            continue
        data = json.loads(mj.read_text())
        c = data["config"]
        g = group_of(d.name)
        va = data["val"]["auroc"]
        n += 1
        if g not in best or va > best[g][0]:
            best[g] = (va, {
                "experiment": g, "layer": c["layer_tag"], "agg": c["agg_method"],
                "ev": c["ev"], "score": c["score_method"], "drop_k": c["drop_k"],
                "pca_k": data.get("pca_k"),
                "val_auroc": round(data["val"]["auroc"], 4), "test_auroc": round(data["test"]["auroc"], 4),
                "val_aupr": round(data["val"]["aupr"], 4), "test_aupr": round(data["test"]["aupr"], 4),
            })
    rows = [best[g][1] for g in sorted(best)]
    return rows, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True,
                    help="fnmatch over grid dir names, e.g. 'grid_example_*'")
    ap.add_argument("--results_root", type=Path, default=ROOT)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rows, n = pick_best_rows(args.results_root, args.pattern)
    print(f"scanned {n} configs → {len(rows)} groups\n")
    for r in rows:
        print(f"  {r['experiment']:<42} {r['layer']:<4} {r['agg']:<6} EV{r['ev']:<5} {r['score']:<14} "
              f"dk{r['drop_k']:<3} val={r['val_auroc']:.4f} test={r['test_auroc']:.4f}")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
