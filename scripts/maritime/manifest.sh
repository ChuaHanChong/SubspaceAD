#!/bin/bash
# manifest.sh — declarative experiment tables for the maritime study (sourced).
#
# Reproduces every experiment the old per-family drivers covered:
#   run_all.sh    : 5 local caches + 2 fuse caches; 8 maritime grids x 5 sizes
#   run_rgbirX.sh : grid_rgbirX x 5 sizes (uses the origRGB+degIR fuse cache)
#   run_infiray.sh: 6 infiray caches; 18 grids (2 bb x 3 fit x 3 eval), --size all
#   run_hf.sh     : 3 HF caches; grid_hf_{s,b,l} x 5 sizes
#   run_infcross.sh: 8 cross_grid experiments
#
# Requires lib.sh to have been sourced first (uses CACHE, *_ROOT, CKPT_*, GRIDS, SIZES).
#
# Row formats:
#   CACHES : "name|builder|k=v ..."   builder in {local,fuse,infiray,hf}
#   GRIDS  : "base_prefix|feat|eval|root|sizes|family"
#            sizes = "$SIZES" (5 sweep) or "all"; eval = "-" for same-cache.
#   CROSS  : bare experiment code for cross_grid.py --experiment

# ====================================================================== CACHES
# local  -> dump_chain.sh  (env: DATA_ROOT, OUT_NAME, CKPT, GPU, CHECK_DEGRADED)
# fuse   -> fuse_pool_features.py  (--cache_a/--cache_b/--tag_a/--tag_b/--out_cache_file)
# infiray-> extract_infiray.py     (--backbone/--variant/--out_cache_file)
# hf     -> extract_features_hf.py  (--model_ckpt/--data_root/--out_cache_file)
CACHES=(
  # --- maritime single-modality (orig + continual backbones) ---
  "rgb.npz|local|root=$RGB_ROOT ckpt=$CKPT_ORIG gate=0"
  "ir.npz|local|root=$IR_ROOT ckpt=$CKPT_ORIG gate=0"
  "ir_degraded.npz|local|root=$DEG_ROOT ckpt=$CKPT_ORIG gate=1"
  "ir_continual.npz|local|root=$IR_ROOT ckpt=$CKPT_CONT gate=0"
  "ir_degraded_continual.npz|local|root=$DEG_ROOT ckpt=$CKPT_CONT gate=1"
  # --- maritime pooled (sample-axis fuse) ---
  "rgbir_pool.npz|fuse|a=rgb.npz b=ir.npz tag_a=rgb tag_b=ir"
  "rgbir_pool_origRGB_degIR.npz|fuse|a=rgb.npz b=ir_degraded.npz tag_a=rgb tag_b=irdeg"
  # --- Infiray (2 backbones x 3 variants) ---
  "infiray_base_raw.npz|infiray|backbone=base variant=raw"
  "infiray_base_enhwo.npz|infiray|backbone=base variant=enhwo"
  "infiray_base_enhw.npz|infiray|backbone=base variant=enhw"
  "infiray_continual_raw.npz|infiray|backbone=continual variant=raw"
  "infiray_continual_enhwo.npz|infiray|backbone=continual variant=enhwo"
  "infiray_continual_enhw.npz|infiray|backbone=continual variant=enhw"
  # --- off-the-shelf HF DINOv2 (degraded IR) ---
  "hf_s_degraded.npz|hf|ckpt=facebook/dinov2-small"
  "hf_b_degraded.npz|hf|ckpt=facebook/dinov2-base"
  "hf_l_degraded.npz|hf|ckpt=facebook/dinov2-large"
)

# ======================================================================= GRIDS
# base_prefix | feat | eval | root | sizes | family
GRIDS_TABLE=(
  # --- 8 maritime grids (run_all.sh EXPERIMENTS) ---
  "grid_rgb|$CACHE/rgb.npz|-|$RGB_ROOT|$SIZES|maritime"
  "grid_ir|$CACHE/ir.npz|-|$IR_ROOT|$SIZES|maritime"
  "grid_irX|$CACHE/ir.npz|$CACHE/ir_degraded.npz|$IR_ROOT|$SIZES|maritime"
  "grid_irdeg|$CACHE/ir_degraded.npz|-|$DEG_ROOT|$SIZES|maritime"
  "grid_rgbir|$CACHE/rgbir_pool.npz|-|$RGB_ROOT|$SIZES|maritime"
  "grid_ircont|$CACHE/ir_continual.npz|-|$IR_ROOT|$SIZES|maritime"
  "grid_irXcont|$CACHE/ir_continual.npz|$CACHE/ir_degraded_continual.npz|$IR_ROOT|$SIZES|maritime"
  "grid_irdegcont|$CACHE/ir_degraded_continual.npz|-|$DEG_ROOT|$SIZES|maritime"
  # --- pooled origRGB + degIR (run_rgbirX.sh) ---
  "grid_rgbirX|$CACHE/rgbir_pool_origRGB_degIR.npz|-|$RGB_ROOT|$SIZES|rgbirX"
  # --- 18 Infiray grids (run_infiray.sh): bb x fit x eval, --size all ---
  "grid_infiray_base_fit-raw_eval-raw|$CACHE/infiray_base_raw.npz|-|$ROOTA|all|infiray"
  "grid_infiray_base_fit-raw_eval-enhwo|$CACHE/infiray_base_raw.npz|$CACHE/infiray_base_enhwo.npz|$ROOTA|all|infiray"
  "grid_infiray_base_fit-raw_eval-enhw|$CACHE/infiray_base_raw.npz|$CACHE/infiray_base_enhw.npz|$ROOTA|all|infiray"
  "grid_infiray_base_fit-enhwo_eval-raw|$CACHE/infiray_base_enhwo.npz|$CACHE/infiray_base_raw.npz|$ROOTA|all|infiray"
  "grid_infiray_base_fit-enhwo_eval-enhwo|$CACHE/infiray_base_enhwo.npz|-|$ROOTA|all|infiray"
  "grid_infiray_base_fit-enhwo_eval-enhw|$CACHE/infiray_base_enhwo.npz|$CACHE/infiray_base_enhw.npz|$ROOTA|all|infiray"
  "grid_infiray_base_fit-enhw_eval-raw|$CACHE/infiray_base_enhw.npz|$CACHE/infiray_base_raw.npz|$ROOTA|all|infiray"
  "grid_infiray_base_fit-enhw_eval-enhwo|$CACHE/infiray_base_enhw.npz|$CACHE/infiray_base_enhwo.npz|$ROOTA|all|infiray"
  "grid_infiray_base_fit-enhw_eval-enhw|$CACHE/infiray_base_enhw.npz|-|$ROOTA|all|infiray"
  "grid_infiray_continual_fit-raw_eval-raw|$CACHE/infiray_continual_raw.npz|-|$ROOTA|all|infiray"
  "grid_infiray_continual_fit-raw_eval-enhwo|$CACHE/infiray_continual_raw.npz|$CACHE/infiray_continual_enhwo.npz|$ROOTA|all|infiray"
  "grid_infiray_continual_fit-raw_eval-enhw|$CACHE/infiray_continual_raw.npz|$CACHE/infiray_continual_enhw.npz|$ROOTA|all|infiray"
  "grid_infiray_continual_fit-enhwo_eval-raw|$CACHE/infiray_continual_enhwo.npz|$CACHE/infiray_continual_raw.npz|$ROOTA|all|infiray"
  "grid_infiray_continual_fit-enhwo_eval-enhwo|$CACHE/infiray_continual_enhwo.npz|-|$ROOTA|all|infiray"
  "grid_infiray_continual_fit-enhwo_eval-enhw|$CACHE/infiray_continual_enhwo.npz|$CACHE/infiray_continual_enhw.npz|$ROOTA|all|infiray"
  "grid_infiray_continual_fit-enhw_eval-raw|$CACHE/infiray_continual_enhw.npz|$CACHE/infiray_continual_raw.npz|$ROOTA|all|infiray"
  "grid_infiray_continual_fit-enhw_eval-enhwo|$CACHE/infiray_continual_enhw.npz|$CACHE/infiray_continual_enhwo.npz|$ROOTA|all|infiray"
  "grid_infiray_continual_fit-enhw_eval-enhw|$CACHE/infiray_continual_enhw.npz|-|$ROOTA|all|infiray"
  # --- 3 HF grids (run_hf.sh): deg->deg, same-cache ---
  "grid_hf_s|$CACHE/hf_s_degraded.npz|-|$DEG_ROOT|$SIZES|hf"
  "grid_hf_b|$CACHE/hf_b_degraded.npz|-|$DEG_ROOT|$SIZES|hf"
  "grid_hf_l|$CACHE/hf_l_degraded.npz|-|$DEG_ROOT|$SIZES|hf"
)

# ======================================================================= CROSS
# cross_grid.py --experiment <code>  (all 8 experiments in cross_grid.py's EXP map;
# the old run_infcross.sh drove 7 of these, and infcross_rgbirX_full.csv shows the
# 8th was also produced — the dual-report tolerates whichever full.csv files exist).
CROSS=( ir irX irdeg rgbir rgbirX ircont irXcont irdegcont )
