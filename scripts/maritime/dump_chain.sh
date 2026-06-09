#!/bin/bash
# Generic single-cache extraction chain (one dataset, one checkpoint, one GPU).
#
# Drives extract_features.py over the 5 incremental sizes (100→10000), building
# one .npz cache. Parameterized by env vars so the same script serves every
# (dataset, model) combination — clean IR, degraded IR, either checkpoint.
#
# Required env:
#   DATA_ROOT   dataset root (clean or degraded)
#   OUT_NAME    cache filename under CACHE_DIR (e.g. ir_continual.npz)
# Optional env:
#   CKPT        checkpoint (default: original RGB-pretrained ViT-L/16)
#   GPU         CUDA device (default 0)
#   CHECK_DEGRADED   if 1, assert the degraded root is complete before extracting
#   CACHE_DIR   output dir (default /data/hanchong/subspacead_cache)
#
# Example:
#   DATA_ROOT=/data/.../klein CKPT=/data/.../Continual-IR/.../teacher_checkpoint.pth \
#   OUT_NAME=ir_continual.npz GPU=1 bash scripts/maritime/dump_chain.sh

set -euo pipefail
cd /home/hcchua/SubspaceAD

PY=/data/hanchong/miniconda3/envs/subspacead/bin/python
CONFIG=/home/hcchua/Maritime-Vessel-Recognition/submodules/dinov2/dinov2/configs/train/vitl16_short.yaml
SUBMOD=/home/hcchua/Maritime-Vessel-Recognition/submodules/dinov2

CKPT=${CKPT:-/data/hanchong/artifacts-dinov2-all/pretraining/ViT-L-16/eval/training_2348399/teacher_checkpoint.pth}
GPU=${GPU:-0}
CACHE_DIR=${CACHE_DIR:-/data/hanchong/subspacead_cache}
CHECK_DEGRADED=${CHECK_DEGRADED:-0}
SIZES="100 500 1000 5000 10000"

: "${DATA_ROOT:?set DATA_ROOT}"
: "${OUT_NAME:?set OUT_NAME}"
mkdir -p "$CACHE_DIR"
cache_file="$CACHE_DIR/$OUT_NAME"
log="$CACHE_DIR/${OUT_NAME%.npz}.extract.log"

if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint not found: $CKPT"; exit 1; fi

# Optional completeness gate (for degraded roots that may still be generating).
if [ "$CHECK_DEGRADED" = "1" ]; then
  echo "=== checking degraded root completeness: $DATA_ROOT ==="
  ood_n=$(find "$DATA_ROOT/Out-of-distribution" -name '*.jpg' 2>/dev/null | wc -l)
  train_classes=$(ls "$DATA_ROOT/In-distribution_10000perCat/train" 2>/dev/null | wc -l)
  miss=0
  for size in $SIZES; do
    [ -d "$DATA_ROOT/In-distribution_${size}perCat" ] || { echo "  MISSING In-distribution_${size}perCat"; miss=1; }
  done
  echo "  OOD jpg: $ood_n (expect 800)   train classes: $train_classes (expect 17)"
  if [ "$ood_n" -lt 800 ] || [ "$train_classes" -lt 17 ] || [ "$miss" -ne 0 ]; then
    echo "ERROR: degraded root incomplete — aborting."; exit 1
  fi
  echo "  OK — complete."
fi

echo "=== extracting $OUT_NAME  (GPU $GPU, ckpt=$(basename "$(dirname "$(dirname "$CKPT")")"))  ==="
: > "$log"
for size in $SIZES; do
  echo "[GPU $GPU] $OUT_NAME ${size}perCat (incremental)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/maritime/extract_features.py \
    --data_root "$DATA_ROOT" \
    --in_dist_subset "In-distribution_${size}perCat" \
    --size "$size" \
    --id_test_subset In-distribution \
    --ood_subset Out-of-distribution \
    --config_file "$CONFIG" \
    --pretrained_weights "$CKPT" \
    --local_submodule_path "$SUBMOD" \
    --local_norm_mean 0.5 0.5 0.5 \
    --local_norm_std  0.5 0.5 0.5 \
    --image_res 224 \
    --batch_size 16 \
    --out_cache_file "$cache_file" \
    student.arch=vit_large \
    student.block_chunks=4 \
    student.num_register_tokens=4 \
    student.interpolate_antialias=true \
    student.interpolate_offset=0.0 \
    >> "$log" 2>&1
  echo "[GPU $GPU] $OUT_NAME ${size}perCat done"
done

echo "==== $OUT_NAME complete ===="
du -sh "$cache_file"
