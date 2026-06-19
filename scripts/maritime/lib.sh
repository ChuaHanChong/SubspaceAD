#!/bin/bash
# lib.sh — shared library for the maritime per-stage run scripts (sourced, not run).
#
# Provides:
#   - shared environment (PY, CACHE, RESULTS, GRIDS, TABLES, SIZES, dataset roots,
#     checkpoints, Infiray root) — taken verbatim from the old per-family drivers.
#   - run_queue(): the canonical flock 4-GPU work queue (byte-identical to the one
#     in the old run_infiray/hf/infcross/rgbirX drivers).
#   - skip_if_exists(): skip-if-the-output-cache-is-present helper.
#   - DRY support: runq()/run() honor DRY=1 by ECHOING the exact command prefixed
#     'DRYCMD: ' to stdout instead of executing (so a verifier can diff the
#     command set the stage would have emitted).
#
# Stage scripts: `source "$(dirname "$0")/lib.sh"` (then manifest.sh), from repo root.

set -uo pipefail
cd /home/hcchua/SubspaceAD

# ---- shared environment ----
PY=/data/hanchong/miniconda3/envs/subspacead/bin/python
CACHE=/data/hanchong/subspacead_cache
RESULTS=/home/hcchua/SubspaceAD/results_maritime
GRIDS=$RESULTS/grids
TABLES=$RESULTS/tables
SIZES="100 500 1000 5000 10000"

# dataset roots
RGB_ROOT=/data/hanchong/images-splitted-3
IR_ROOT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein
DEG_ROOT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein-degraded
# Infiray cross-dataset root (val source; 红外船舶数据库)
ROOTA=/data/hanchong/other-infrared-datasets/processed-data/红外船舶数据库

# SSL checkpoints
CKPT_ORIG=/data/hanchong/artifacts-dinov2-all/pretraining/ViT-L-16/eval/training_2348399/teacher_checkpoint.pth
CKPT_CONT=/data/hanchong/artifacts-dinov2-all/pretraining/ViT-L-16-Continual-IR/eval/training_51199/teacher_checkpoint.pth

mkdir -p "$CACHE" "$GRIDS" "$TABLES"

# DRY: when set to 1, runq/run echo the command (prefixed 'DRYCMD: ') instead of
# executing. Default off.
DRY=${DRY:-0}

# ---- generic 4-GPU flock queue (verbatim from run_infiray.sh) ----
run_queue() {  # reads "cmd" lines from $1
  local qf=$1 lock="$1.lock"; : > "$lock"
  pop(){ exec {fd}<"$lock"; flock "$fd"; local l; l=$(head -n1 "$qf"); [ -n "$l" ] && sed -i '1d' "$qf"; flock -u "$fd"; exec {fd}<&-; printf '%s' "$l"; }
  worker(){ local g=$1 l; while :; do l=$(pop); [ -z "$l" ] && break; eval "CUDA_VISIBLE_DEVICES=$g $l"; done; }
  for g in 0 1 2 3; do worker "$g" & done; wait; rm -f "$qf" "$lock"
}

# ---- helpers ----

# skip_if_exists FILE LABEL  -> returns 0 (skip) if FILE exists, else 1 (proceed).
# In DRY mode never skips, so the verifier sees the full intended command set.
skip_if_exists() {
  local f=$1 label=${2:-$1}
  if [ "$DRY" != "1" ] && [ -f "$f" ]; then
    echo "skip $label (exists: $f)"
    return 0
  fi
  return 1
}

# runq QUEUEFILE CMD...  -> append CMD to the queue (executed later by run_queue).
# In DRY mode, echo 'DRYCMD: CMD' to stdout and do NOT enqueue.
runq() {
  local qf=$1; shift
  if [ "$DRY" = "1" ]; then
    echo "DRYCMD: $*"
  else
    echo "$*" >> "$qf"
  fi
}

# run CMD...  -> run CMD directly (foreground). In DRY mode echo 'DRYCMD: CMD'.
run() {
  if [ "$DRY" = "1" ]; then
    echo "DRYCMD: $*"
  else
    eval "$*"
  fi
}
