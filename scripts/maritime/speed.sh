#!/bin/bash
# speed.sh — compute-speed benchmark for DINOv2 ViT-S/B/L (+ PCA grid timing).
# Writes tables/speed_bench.csv.  DRY=1 echoes the command.
#
#   speed.sh [--only ...]   (--only accepted for symmetry; single benchmark)

HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck source=lib.sh
source "$HERE/lib.sh"

while [ $# -gt 0 ]; do
  case "$1" in
    --only) shift 2;;
    *) echo "speed.sh: unknown arg '$1'" >&2; exit 2;;
  esac
done

run "CUDA_VISIBLE_DEVICES=0 $PY scripts/maritime/bench_speed.py || true"
