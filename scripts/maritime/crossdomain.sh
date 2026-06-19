#!/bin/bash
# crossdomain.sh — Ask-2: maritime-fit detectors grid-evaluated on Infiray.
#
# Phase A: cross_grid.py --experiment <e> for every CROSS entry, across 4 GPUs,
#          writing grids/infcross_<e>_full.csv.
# Phase B: infcross_report.py emits the two reports (Variant A maritime-config +
#          Variant B infiray-val-select) -> tables/infcross_maritimecfg.csv +
#          tables/infcross_valselect.csv.
#
#   crossdomain.sh [--only ...]   (--only is accepted for interface symmetry but
#                                  there is a single cross-domain family)
#   DRY=1 echoes the commands.

HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck source=lib.sh
source "$HERE/lib.sh"
# shellcheck source=manifest.sh
source "$HERE/manifest.sh"

while [ $# -gt 0 ]; do
  case "$1" in
    --only) shift 2;;   # single family; accepted and ignored
    *) echo "crossdomain.sh: unknown arg '$1'" >&2; exit 2;;
  esac
done

# ---- Phase A: per-experiment cross-domain grids ----
Q=$(mktemp)
for e in "${CROSS[@]}"; do
  runq "$Q" "$PY scripts/maritime/cross_grid.py --experiment $e > $GRIDS/infcross_${e}.log 2>&1"
done
if [ "$DRY" != "1" ]; then run_queue "$Q"; else rm -f "$Q"; fi

# ---- Phase B: dual report from the per-experiment full CSVs ----
run "$PY scripts/maritime/infcross_report.py $GRIDS $TABLES"
