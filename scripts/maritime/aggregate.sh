#!/bin/bash
# aggregate.sh — summarize the grids into the study's CSV tables.
#
#   maritime : aggregate.py --pattern 'grid_*' --all   -> tables/grid_summary_all.csv
#              (the --all dump's strict regex keeps only the 8 maritime datasets,
#               reproducing the former aggregate_results.py output exactly.)
#   infiray  : aggregate.py --pattern 'grid_infiray_*' -> tables/infiray_grid_best.csv (val-best)
#   hf       : aggregate.py --pattern 'grid_hf_*'      -> tables/hf_grid_best.csv      (val-best)
#   balanced : eval_balanced.py                        -> tables/balanced_test.csv
#
#   aggregate.sh [--only maritime|infiray|hf|balanced]    DRY=1 echoes commands.

HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck source=lib.sh
source "$HERE/lib.sh"
# shellcheck source=manifest.sh
source "$HERE/manifest.sh"

ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY=$2; shift 2;;
    *) echo "aggregate.sh: unknown arg '$1'" >&2; exit 2;;
  esac
done

agg_maritime() {
  run "$PY scripts/maritime/aggregate.py --pattern 'grid_*' --all --results_root $GRIDS --csv $TABLES/grid_summary_all.csv || true"
}
agg_infiray() {
  run "$PY scripts/maritime/aggregate.py --pattern 'grid_infiray_*' --results_root $GRIDS --csv $TABLES/infiray_grid_best.csv || true"
}
agg_hf() {
  run "$PY scripts/maritime/aggregate.py --pattern 'grid_hf_*' --results_root $GRIDS --csv $TABLES/hf_grid_best.csv || true"
}
agg_balanced() {
  run "$PY scripts/maritime/eval_balanced.py || true"
}

case "$ONLY" in
  "")        agg_maritime; agg_infiray; agg_hf; agg_balanced;;
  maritime)  agg_maritime;;
  infiray)   agg_infiray;;
  hf)        agg_hf;;
  balanced)  agg_balanced;;
  *) echo "aggregate.sh: unknown family '$ONLY' (maritime|infiray|hf|balanced)" >&2; exit 2;;
esac
