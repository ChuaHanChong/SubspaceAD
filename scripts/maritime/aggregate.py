"""Aggregate maritime / Infiray / HF grid result dirs.

Merges the former `pick_best.py` (generic val-best per group) and
`aggregate_results.py` (full grid dump + per-axis slice report) into one CLI.

Modes:
  default  val-best (max val.auroc) per GROUP, where group = dir name with the
           trailing config tag stripped. Reads each dir's metrics.json 'config'.
  --all    one row per config (full grid dump, reproduces grid_summary_all.csv).
  --detail per-axis slice report to stdout (val-best config + per-axis slices).

Examples:
  python aggregate.py --pattern 'grid_infiray_*' --csv results_maritime/tables/infiray_grid_best.csv
  python aggregate.py --pattern 'grid_*' --all --csv results_maritime/tables/grid_summary_all.csv
  python aggregate.py --pattern 'grid_*' --detail
"""
import argparse
import csv
import fnmatch
import json
import re
from pathlib import Path

ROOT = Path("/home/hcchua/SubspaceAD/results_maritime/grids")

# --- val-best-per-group (from pick_best.py) -------------------------------
# Trailing config tag: _<layer>_<m|c>_EV####_<score>_dk#
TAG = re.compile(r"_(?:L\d+|Lmid)_[mc]_EV\d{4}_(?:rec|mah|cos|euc)_dk\d+$")


def group_of(name: str) -> str:
    return TAG.sub("", name)


# --- full grid dump (from aggregate_results.py) ---------------------------
# Filename pattern: grid_{dataset}_{size}_{layer}_{agg}_EV{nnnn}_{score}_dk{k}
# dataset alternation lists longer prefixes first so e.g. "irdeg" is not
# shadowed by "ir". Base model: rgb, ir (original), irX (fit-original/eval-deg),
# irdeg (deg/deg), rgbir (pooled). Continual model adds the *cont variants:
# ircont (original/original), irXcont (fit-original/eval-deg), irdegcont (deg/deg).
PATTERN = re.compile(
    r"^grid_(?P<dataset>rgbir|irdegcont|irXcont|ircont|irdeg|irX|rgb|ir)_(?P<size>\d+)"
    r"_(?P<layer>L1|L2|L4|L6|L8|L12|L18|Lmid)"
    r"_(?P<agg>m|c)"
    r"_EV(?P<ev>\d{4})"
    r"_(?P<score>rec|mah|cos|euc)"
    r"_dk(?P<dk>\d+)$"
)
SCORE_LONG = {"rec": "reconstruction", "mah": "mahalanobis",
              "cos": "cosine", "euc": "euclidean"}
AGG_LONG = {"m": "mean", "c": "concat"}

AXES = ["size", "layer_tag", "agg", "ev", "score", "drop_k"]
LAYER_ORDER = ["L1", "L2", "L4", "L6", "L8", "L12", "L18", "Lmid"]
SCORE_ORDER = ["reconstruction", "mahalanobis", "cosine", "euclidean"]
AGG_ORDER = ["mean", "concat"]


# ==========================================================================
# default mode: val-best per group
# ==========================================================================
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
                "test_fpr95": round(data["test"]["fpr_at_95tpr"], 4),
            })

    rows = [best[g][1] for g in sorted(best)]
    return rows, n


def run_pick_best(args):
    rows, n = pick_best_rows(args.results_root, args.pattern)
    print(f"scanned {n} configs → {len(rows)} groups\n")
    for r in rows:
        print(f"  {r['experiment']:<42} {r['layer']:<4} {r['agg']:<6} EV{r['ev']:<5} {r['score']:<14} "
              f"dk{r['drop_k']:<3} val={r['val_auroc']:.4f} test={r['test_auroc']:.4f}")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")


# ==========================================================================
# --all mode: full grid dump
# ==========================================================================
def discover_rows(results_root: Path, pattern: str) -> list[dict]:
    rows: list[dict] = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        if not fnmatch.fnmatch(d.name, pattern):
            continue
        m = PATTERN.match(d.name)
        if not m:
            continue
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text())
        rows.append({
            "dataset":   m.group("dataset"),
            "size":      int(m.group("size")),
            "layer_tag": m.group("layer"),
            "agg":       AGG_LONG[m.group("agg")],
            "ev":        int(m.group("ev")) / 1000.0,
            "score":     SCORE_LONG[m.group("score")],
            "drop_k":    int(m.group("dk")),
            "val_auroc":  data["val"]["auroc"],
            "test_auroc": data["test"]["auroc"],
            "val_aupr":   data["val"]["aupr"],
            "test_aupr":  data["test"]["aupr"],
            "val_fpr95":  data["val"]["fpr_at_95tpr"],
            "test_fpr95": data["test"]["fpr_at_95tpr"],
            "pca_k": int(data["pca_k"]),
            "dirname": d.name,
        })
    return rows


def write_csv(rows: list[dict], out_path: Path):
    cols = ["dataset", "size", "layer_tag", "agg", "ev", "score", "drop_k",
            "val_auroc", "test_auroc", "val_aupr", "test_aupr",
            "val_fpr95", "test_fpr95", "pca_k", "dirname"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"wrote {out_path} ({len(rows)} rows)")


# ==========================================================================
# --detail mode: per-axis slice report
# ==========================================================================
def _print_axis_summary(rows: list[dict], axis_key: str, axis_label: str, fixed_label: str):
    """Print AUROC per axis value, marking val-best (★) and test-best (▲)."""
    print(f"{axis_label:<14}  {'val_AUROC':>10}  {'test_AUROC':>10}    {fixed_label}")
    val_best  = max(rows, key=lambda r: r["val_auroc"])
    test_best = max(rows, key=lambda r: r["test_auroc"])
    for r in rows:
        mark = ""
        if r is val_best:  mark += " ★val"
        if r is test_best: mark += " ▲test"
        print(f"  {str(r[axis_key]):<12}  {r['val_auroc']:>10.4f}  {r['test_auroc']:>10.4f}{mark}")
    print()


def _sort_key(axis: str):
    if axis == "layer_tag":
        return lambda r: LAYER_ORDER.index(r["layer_tag"])
    if axis == "score":
        return lambda r: SCORE_ORDER.index(r["score"])
    if axis == "agg":
        return lambda r: AGG_ORDER.index(r["agg"])
    return lambda r: r[axis]


def report_dataset(rows_all: list[dict], dataset: str):
    rows = [r for r in rows_all if r["dataset"] == dataset]
    if not rows:
        print(f"(no {dataset.upper()} results)")
        return None

    print("=" * 78)
    print(f"=== {dataset.upper()}  —  {len(rows)} configs ===")
    print("=" * 78)

    val_best = max(rows, key=lambda r: r["val_auroc"])
    print(
        f"\n★ Val-best config: size={val_best['size']} {val_best['layer_tag']} "
        f"agg={val_best['agg']} EV={val_best['ev']:.3f} "
        f"{val_best['score']} drop_k={val_best['drop_k']}"
    )
    print(
        f"  val  AUROC={val_best['val_auroc']:.4f}  AUPR={val_best['val_aupr']:.4f}  "
        f"FPR95={val_best['val_fpr95']:.4f}"
    )
    print(
        f"  test AUROC={val_best['test_auroc']:.4f}  AUPR={val_best['test_aupr']:.4f}  "
        f"FPR95={val_best['test_fpr95']:.4f}  (← reported headline)"
    )
    print(f"  pca_k={val_best['pca_k']}")
    print()

    # Per-axis slices: hold the OTHER 6 axes at val-best, vary one
    def fixed_filter(varied: str) -> list[dict]:
        return [r for r in rows if all(r[k] == val_best[k] for k in AXES if k != varied)]

    for axis in AXES:
        sub = fixed_filter(axis)
        if len(sub) < 2:
            continue  # nothing to compare
        sub = sorted(sub, key=_sort_key(axis))
        fixed_desc = " ".join(
            f"{k}={val_best[k]}" for k in AXES if k != axis
        )
        print(f"-- {axis} slice (others held at val-best) --")
        _print_axis_summary(sub, axis, axis, fixed_desc)

    return val_best


def run_detail(args):
    rows = discover_rows(args.results_root, args.pattern)
    print(f"Discovered {len(rows)} grid-result rows in {args.results_root}\n")

    # Report every dataset/experiment present, in a stable, readable order.
    order = ["rgb", "ir", "irX", "irdeg", "rgbir",
             "ircont", "irXcont", "irdegcont"]
    present = [ds for ds in order if any(r["dataset"] == ds for r in rows)]
    present += sorted({r["dataset"] for r in rows} - set(present))

    bests = {}
    for ds in present:
        bests[ds] = report_dataset(rows, ds)
        print()

    # Cross-experiment headline comparison (val-best of each).
    if len(present) > 1:
        print("=" * 78)
        print("=== Cross-experiment summary (each at its own val-best config) ===")
        print("=" * 78)
        label = {"rgb": "Base · RGB · original→original", "ir": "Base · IR · original→original",
                 "irX": "Base · IR · original→degraded", "irdeg": "Base · IR · degraded→degraded",
                 "rgbir": "Base · RGB+IR pooled · original→original",
                 "ircont": "Continual · IR · original→original",
                 "irXcont": "Continual · IR · original→degraded",
                 "irdegcont": "Continual · IR · degraded→degraded"}
        def _summary(b: dict) -> str:
            return (f"size={b['size']:<5} {b['layer_tag']:<4} agg={b['agg']:<6} "
                    f"EV={b['ev']:.3f} {b['score']:<14} "
                    f"dk={b['drop_k']:<3} val={b['val_auroc']:.4f} test={b['test_auroc']:.4f}")
        for ds in present:
            if bests[ds] is not None:
                print(f"{label.get(ds, ds):<28} {_summary(bests[ds])}")

    if args.csv is not None:
        write_csv(rows, args.csv)


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True,
                    help="fnmatch over dir names, e.g. 'grid_infiray_*' or 'grid_*'")
    ap.add_argument("--results_root", type=Path, default=ROOT)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--all", action="store_true",
                    help="one row per config (full grid dump)")
    ap.add_argument("--detail", action="store_true",
                    help="per-axis slice report to stdout")
    args = ap.parse_args()

    if args.detail:
        run_detail(args)
    elif args.all:
        rows = discover_rows(args.results_root, args.pattern)
        print(f"Discovered {len(rows)} grid-result rows in {args.results_root}\n")
        if args.csv is not None:
            write_csv(rows, args.csv)
    else:
        run_pick_best(args)


if __name__ == "__main__":
    main()
