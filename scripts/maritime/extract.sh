#!/bin/bash
# extract.sh — build every feature cache in the manifest (skip-if-present).
#
# Dispatches each CACHES entry by its builder:
#   local  -> dump_chain.sh        (env: DATA_ROOT/OUT_NAME/CKPT/GPU/CHECK_DEGRADED/CACHE_DIR)
#   fuse   -> fuse_pool_features.py (numpy sample-axis merge; no GPU)
#   infiray-> extract_infiray.py    (one cache per backbone x variant; 4-GPU queue)
#   hf     -> extract_features_hf.py(one model per GPU)
#
#   extract.sh [--only local|fuse|infiray|hf]      DRY=1 echoes commands.
#
# GPU-bound local builds (dump_chain) are pinned per build because dump_chain
# re-sets CUDA_VISIBLE_DEVICES internally; they run backgrounded, round-robin
# over GPUs 0..3, mirroring the old run_all.sh layout.

HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck source=lib.sh
source "$HERE/lib.sh"
# shellcheck source=manifest.sh
source "$HERE/manifest.sh"

ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY=$2; shift 2;;
    *) echo "extract.sh: unknown arg '$1'" >&2; exit 2;;
  esac
done

# parse "k=v k=v" payload into the named bash vars (k must be a valid identifier)
_kv() {  # _kv "a=1 b=2" -> sets $a $b
  local tok k v
  for tok in $1; do k=${tok%%=*}; v=${tok#*=}; printf -v "$k" '%s' "$v"; done
}

build_local() {  # name "k=v..." gpu
  local name=$1 kv=$2 gpu=$3 root="" ckpt="" gate=""
  _kv "$kv"
  if skip_if_exists "$CACHE/$name" "$name"; then return; fi
  # dump_chain.sh self-logs to $CACHE/${name%.npz}.extract.log; match run_all.sh
  # build_cache (no extra stdout redirect).
  run "DATA_ROOT=$root CKPT=$ckpt OUT_NAME=$name GPU=$gpu CHECK_DEGRADED=$gate CACHE_DIR=$CACHE bash scripts/maritime/dump_chain.sh"
}

build_fuse() {  # name "k=v..."
  local name=$1 kv=$2 a="" b="" tag_a="" tag_b=""
  _kv "$kv"
  if skip_if_exists "$CACHE/$name" "$name"; then return; fi
  run "$PY scripts/maritime/fuse_pool_features.py --cache_a $CACHE/$a --cache_b $CACHE/$b --tag_a $tag_a --tag_b $tag_b --out_cache_file $CACHE/$name"
}

# returns the queued-command string for one infiray cache (or empty if skipped)
queue_infiray() {  # name "k=v..." -> echoes cmd
  local name=$1 kv=$2 backbone="" variant=""
  _kv "$kv"
  if skip_if_exists "$CACHE/$name" "$name" >&2; then return; fi
  echo "$PY scripts/maritime/extract_infiray.py --backbone $backbone --variant $variant --out_cache_file $CACHE/$name >> $CACHE/${name%.npz}.log 2>&1"
}

# ---- local: backgrounded, round-robin GPUs (skip-if-present returns instantly) ----
run_local() {
  local gpu=0 e name builder kv
  for e in "${CACHES[@]}"; do
    IFS='|' read -r name builder kv <<< "$e"
    [ "$builder" = "local" ] || continue
    ( build_local "$name" "$kv" "$gpu" ) &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
}

# ---- fuse: pure numpy, sequential (needs the single-modality caches present) ----
run_fuse() {
  local e name builder kv
  for e in "${CACHES[@]}"; do
    IFS='|' read -r name builder kv <<< "$e"
    [ "$builder" = "fuse" ] || continue
    build_fuse "$name" "$kv"
  done
}

# ---- infiray: 6 caches through the 4-GPU flock queue ----
run_infiray() {
  local Q cmd e name builder kv
  Q=$(mktemp)
  for e in "${CACHES[@]}"; do
    IFS='|' read -r name builder kv <<< "$e"
    [ "$builder" = "infiray" ] || continue
    cmd=$(queue_infiray "$name" "$kv")
    [ -n "$cmd" ] && runq "$Q" "$cmd"
  done
  if [ "$DRY" != "1" ]; then run_queue "$Q"; else rm -f "$Q"; fi
}

# ---- hf: one full extraction per model, one GPU each (backgrounded) ----
run_hf() {
  local gpu=0 e name builder kv ckpt
  for e in "${CACHES[@]}"; do
    IFS='|' read -r name builder kv <<< "$e"
    [ "$builder" = "hf" ] || continue
    ckpt=""; _kv "$kv"
    if skip_if_exists "$CACHE/$name" "$name"; then continue; fi
    ( run "CUDA_VISIBLE_DEVICES=$gpu $PY scripts/maritime/extract_features_hf.py --model_ckpt $ckpt --data_root $DEG_ROOT --out_cache_file $CACHE/$name > $CACHE/${name%.npz}.log 2>&1" ) &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
}

case "$ONLY" in
  "")        run_local; run_fuse; run_infiray; run_hf;;
  local)     run_local;;
  fuse)      run_fuse;;
  infiray)   run_infiray;;
  hf)        run_hf;;
  *) echo "extract.sh: unknown family '$ONLY' (local|fuse|infiray|hf)" >&2; exit 2;;
esac
