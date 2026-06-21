#!/bin/bash
# run.sh — minimal SubspaceAD image-level OOD example.
#
# One generic pipeline: extract DINOv2 CLS features -> PCA hyperparameter grid -> pick the
# val-best. A dataset = ID/OOD image folders; local vs off-the-shelf = a --backbone switch.
# Edit the variables below or pass them as env.
#
#   bash scripts/maritime/run.sh
#   NAME=hf BACKBONE=hf HF_MODEL=facebook/dinov2-small ID_FIT=... OOD_TEST=... bash scripts/maritime/run.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"        # repo root

PY="${PY:-python}"
NAME="${NAME:-example}"                           # output prefix / cache name

# ---- backbone: local DINOv2 checkpoint OR off-the-shelf HF ----
BACKBONE="${BACKBONE:-local}"                     # local | hf
CKPT="${CKPT:-/path/to/dinov2_teacher_checkpoint.pth}"
DINO_CONFIG="${DINO_CONFIG:-/path/to/dinov2/configs/train/vitl16_short.yaml}"
DINO_SUBMOD="${DINO_SUBMOD:-/path/to/dinov2}"
HF_MODEL="${HF_MODEL:-facebook/dinov2-small}"
# local backbone DINOv2 student opts (edit if your checkpoint is a different arch)
LOCAL_OPTS="${LOCAL_OPTS:-student.arch=vit_large student.block_chunks=4 student.num_register_tokens=4 student.interpolate_antialias=true student.interpolate_offset=0.0}"

# ---- dataset: one image folder per split (ID = normal, OOD = anomaly) ----
ID_FIT="${ID_FIT:-/path/to/id/fit}"
ID_VAL="${ID_VAL:-/path/to/id/val}"
ID_TEST="${ID_TEST:-/path/to/id/test}"
OOD_VAL="${OOD_VAL:-/path/to/ood/val}"
OOD_TEST="${OOD_TEST:-/path/to/ood/test}"

CACHE="${CACHE:-cache}"             # feature .npz output dir
GRIDS="${GRIDS:-results/grids}"     # per-config metrics.json output dir
TABLES="${TABLES:-results/tables}"  # val-best CSV output dir
mkdir -p "$CACHE" "$GRIDS" "$TABLES"

CACHE_FILE="$CACHE/$NAME.npz"

# 1) extract features once (ID fit + ID/OOD val/test)
if [ "$BACKBONE" = "hf" ]; then
  "$PY" scripts/maritime/extract_features.py --backbone hf --hf_model "$HF_MODEL" \
    --id_fit "$ID_FIT" --id_val "$ID_VAL" --id_test "$ID_TEST" \
    --ood_val "$OOD_VAL" --ood_test "$OOD_TEST" --out_cache_file "$CACHE_FILE"
else
  # shellcheck disable=SC2086  # LOCAL_OPTS is intentionally word-split into separate args
  "$PY" scripts/maritime/extract_features.py --backbone local \
    --ckpt "$CKPT" --config "$DINO_CONFIG" --submodule "$DINO_SUBMOD" \
    --id_fit "$ID_FIT" --id_val "$ID_VAL" --id_test "$ID_TEST" \
    --ood_val "$OOD_VAL" --ood_test "$OOD_TEST" --out_cache_file "$CACHE_FILE" \
    $LOCAL_OPTS
fi

# 2) hyperparameter grid (PCA fit on all id_fit features)
"$PY" scripts/maritime/anomaly_detection.py \
  --feature_cache_file "$CACHE_FILE" \
  --out_prefix "grid_${NAME}_" --results_root "$GRIDS"

# 3) pick the val-best config
"$PY" scripts/maritime/aggregate.py --pattern "grid_${NAME}_*" \
  --results_root "$GRIDS" --csv "$TABLES/${NAME}_best.csv"
