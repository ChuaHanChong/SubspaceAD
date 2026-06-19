#!/bin/bash
# grid.sh — run the anomaly_detection.py hyperparameter grid for every GRIDS_TABLE
# entry, across all four GPUs via the shared flock queue.
#
#   grid.sh [--only maritime|rgbirX|infiray|hf]     DRY=1 echoes the commands.
#
# Per-family command shape is preserved byte-for-byte from the old drivers:
#   maritime : --in_dist_subset In-distribution_<size>perCat --ood_subset Out-of-distribution
#   rgbirX   : --in_dist_subset rgbirX_<size>                 (no --ood_subset)
#   hf       : --in_dist_subset In-distribution_<size>perCat  (no --ood_subset)
#   infiray  : --in_dist_subset infiray --size all            (no --ood_subset)
# out_prefix tags + stdout log paths match the originals exactly.

HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck source=lib.sh
source "$HERE/lib.sh"
# shellcheck source=manifest.sh
source "$HERE/manifest.sh"

ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY=$2; shift 2;;
    *) echo "grid.sh: unknown arg '$1'" >&2; exit 2;;
  esac
done

Q=$(mktemp)

# enqueue one anomaly_detection.py command (skip-if feat/eval cache missing, like the old drivers)
enqueue_grid() {  # base feat evalc root size family
  local base=$1 feat=$2 evalc=$3 root=$4 size=$5 family=$6
  local out_prefix indist ood log ea

  case "$family" in
    maritime)
      out_prefix="${base}_${size}_"; indist="In-distribution_${size}perCat"
      ood="--ood_subset Out-of-distribution"; log="$GRIDS/${out_prefix}stdout.log";;
    rgbirX)
      out_prefix="${base}_${size}_"; indist="rgbirX_${size}"
      ood=""; log="$GRIDS/${out_prefix}stdout.log";;
    hf)
      out_prefix="${base}_${size}_"; indist="In-distribution_${size}perCat"
      ood=""; log="$GRIDS/${out_prefix}stdout.log";;
    infiray)
      out_prefix="${base}_"; indist="infiray"
      ood=""; log="$GRIDS/${out_prefix}stdout.log";;
    *) echo "grid.sh: unknown family '$family'" >&2; return;;
  esac

  if [ "$DRY" != "1" ]; then
    [ -f "$feat" ] || { echo "skip ${out_prefix} (missing $feat)"; return; }
    [ "$evalc" != "-" ] && { [ -f "$evalc" ] || { echo "skip ${out_prefix} (missing eval $evalc)"; return; }; }
  fi
  ea=""; [ "$evalc" != "-" ] && ea="--eval_cache_file $evalc"

  runq "$Q" "$PY scripts/maritime/anomaly_detection.py --feature_cache_file $feat $ea --data_root $root --in_dist_subset $indist --size $size $ood --out_prefix $out_prefix --results_root $GRIDS > $log 2>&1"
}

for e in "${GRIDS_TABLE[@]}"; do
  IFS='|' read -r base feat evalc root sizes family <<< "$e"
  [ -z "$ONLY" ] || [ "$family" = "$ONLY" ] || continue
  if [ "$sizes" = "all" ]; then
    enqueue_grid "$base" "$feat" "$evalc" "$root" all "$family"
  else
    for s in $sizes; do
      enqueue_grid "$base" "$feat" "$evalc" "$root" "$s" "$family"
    done
  fi
done

if [ "$DRY" != "1" ]; then run_queue "$Q"; else rm -f "$Q"; fi
