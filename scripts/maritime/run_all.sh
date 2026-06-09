#!/bin/bash
# Single orchestrator for the ENTIRE maritime OOD study across 4 GPUs:
# builds every feature cache (skip if present), runs every experiment grid, aggregates.
#
# 8 experiments (2 backbones × conditions):
#   orig:      grid_rgb   RGB clean→clean        grid_ir     IR clean→clean
#              grid_irX   IR clean→degraded      grid_irdeg  IR degraded→degraded
#              grid_rgbir RGB∪IR pooled clean
#   continual: grid_ircont IR clean→clean        grid_irXcont IR clean→degraded
#              grid_irdegcont IR degraded→degraded
#
# Phase 0: build missing caches (dump_chain.sh per cache; fuse for the pool).
# Phase 1: all 8 experiments × 5 sizes = 40 grid jobs through a flock 4-GPU queue.
# Phase 2: aggregate to grid_summary_all.csv.
#
# Caches are reused if present, so warm re-runs are grid-only. Degraded caches are
# gated on the degraded dataset being complete (CHECK_DEGRADED in dump_chain.sh).

set -uo pipefail
cd /home/hcchua/SubspaceAD

PY=/data/hanchong/miniconda3/envs/subspacead/bin/python
CACHE=/data/hanchong/subspacead_cache
RESULTS=/home/hcchua/SubspaceAD/results_maritime
GRIDS=$RESULTS/grids
mkdir -p "$CACHE" "$GRIDS"
SIZES="100 500 1000 5000 10000"

RGB_ROOT=/data/hanchong/images-splitted-3
IR_ROOT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein
DEG_ROOT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein-degraded
CKPT_ORIG=/data/hanchong/artifacts-dinov2-all/pretraining/ViT-L-16/eval/training_2348399/teacher_checkpoint.pth
CKPT_CONT=/data/hanchong/artifacts-dinov2-all/pretraining/ViT-L-16-Continual-IR/eval/training_51199/teacher_checkpoint.pth

############################ PHASE 0: caches ############################
echo "==================== PHASE 0: build missing caches ===================="
# build_cache: gpu name data_root ckpt gate   (skips if the .npz already exists)
build_cache() {
  local gpu=$1 name=$2 root=$3 ckpt=$4 gate=$5
  if [ -f "$CACHE/$name" ]; then echo "[GPU $gpu] $name exists — skip"; return; fi
  DATA_ROOT="$root" CKPT="$ckpt" OUT_NAME="$name" GPU="$gpu" CHECK_DEGRADED="$gate" CACHE_DIR="$CACHE" \
    bash scripts/maritime/dump_chain.sh
}
# 5 single-modality caches across 4 GPUs (GPU 0 takes two; existing ones return instantly).
( build_cache 0 rgb.npz                   "$RGB_ROOT" "$CKPT_ORIG" 0
  build_cache 0 ir_degraded_continual.npz "$DEG_ROOT" "$CKPT_CONT" 1 ) &
( build_cache 1 ir.npz                    "$IR_ROOT"  "$CKPT_ORIG" 0 ) &
( build_cache 2 ir_degraded.npz           "$DEG_ROOT" "$CKPT_ORIG" 1 ) &
( build_cache 3 ir_continual.npz          "$IR_ROOT"  "$CKPT_CONT" 0 ) &
wait
# Pooled cache (needs rgb + ir); pure numpy merge, no GPU.
if [ -f "$CACHE/rgb.npz" ] && [ -f "$CACHE/ir.npz" ] && [ ! -f "$CACHE/rgbir_pool.npz" ]; then
  "$PY" scripts/maritime/fuse_pool_features.py
fi
echo "==================== PHASE 0 complete ===================="

############################ PHASE 1: grids ############################
echo "==================== PHASE 1: 8 experiments × 5 sizes ===================="
grid_job() {  # gpu prefix feat eval root size
  local gpu=$1 prefix=$2 feat=$3 evalc=$4 root=$5 size=$6
  local out_prefix="${prefix}_${size}_"
  local log="$GRIDS/${out_prefix}stdout.log"
  [ -f "$feat" ] || { echo "[GPU $gpu] skip ${out_prefix} (missing $feat)"; return; }
  local ea=()
  [ "$evalc" != "-" ] && { [ -f "$evalc" ] || { echo "[GPU $gpu] skip ${out_prefix} (missing eval $evalc)"; return; }; ea=(--eval_cache_file "$evalc"); }
  echo "[GPU $gpu] grid ${out_prefix}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/maritime/anomaly_detection.py \
    --feature_cache_file "$feat" "${ea[@]}" --data_root "$root" \
    --in_dist_subset "In-distribution_${size}perCat" --size "$size" \
    --ood_subset Out-of-distribution --out_prefix "$out_prefix" \
    --results_root "$GRIDS" > "$log" 2>&1
  echo "[GPU $gpu] done ${out_prefix}"
}

# Experiment table: prefix | feat | eval | root
EXPERIMENTS=(
  "grid_rgb|$CACHE/rgb.npz|-|$RGB_ROOT"
  "grid_ir|$CACHE/ir.npz|-|$IR_ROOT"
  "grid_irX|$CACHE/ir.npz|$CACHE/ir_degraded.npz|$IR_ROOT"
  "grid_irdeg|$CACHE/ir_degraded.npz|-|$DEG_ROOT"
  "grid_rgbir|$CACHE/rgbir_pool.npz|-|$RGB_ROOT"
  "grid_ircont|$CACHE/ir_continual.npz|-|$IR_ROOT"
  "grid_irXcont|$CACHE/ir_continual.npz|$CACHE/ir_degraded_continual.npz|$IR_ROOT"
  "grid_irdegcont|$CACHE/ir_degraded_continual.npz|-|$DEG_ROOT"
)

QUEUE=$(mktemp)
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r prefix feat evalc root <<< "$e"
  for s in $SIZES; do echo "$prefix|$feat|$evalc|$root|$s" >> "$QUEUE"; done
done
# Big sizes first so they don't tail the run.
sort -t'|' -k5 -rn -o "$QUEUE" "$QUEUE"
LOCK="${QUEUE}.lock"; : > "$LOCK"

pop() {  # atomically pop the first queue line
  local line=""
  exec {fd}<"$LOCK"; flock "$fd"
  line=$(head -n1 "$QUEUE"); [ -n "$line" ] && sed -i '1d' "$QUEUE"
  flock -u "$fd"; exec {fd}<&-
  printf '%s' "$line"
}
worker() {
  local gpu=$1 line
  while :; do
    line=$(pop); [ -z "$line" ] && break
    IFS='|' read -r prefix feat evalc root size <<< "$line"
    grid_job "$gpu" "$prefix" "$feat" "$evalc" "$root" "$size"
  done
}
for g in 0 1 2 3; do worker "$g" & done
wait
rm -f "$QUEUE" "$LOCK"
echo "==================== PHASE 1 complete ===================="

############################ PHASE 2: aggregate ############################
echo "==================== PHASE 2: aggregate ===================="
"$PY" scripts/maritime/aggregate_results.py --results_root "$GRIDS" --csv "$RESULTS/grid_summary_all.csv" || true

echo "==== ALL DONE ===="
for prefix in grid_rgb grid_ir grid_irX grid_irdeg grid_rgbir grid_ircont grid_irXcont grid_irdegcont; do
  for size in $SIZES; do
    n=$(ls "$GRIDS/${prefix}_${size}_"*/metrics.json 2>/dev/null | wc -l)
    echo "  ${prefix}_${size}_*: $n / 1920"
  done
done
