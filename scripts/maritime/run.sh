#!/bin/bash
# run.sh — top-level driver for the maritime OOD study.
#
#   run.sh [all|extract|grid|crossdomain|aggregate|speed] [--only FAMILY]
#
# 'all' (default) chains the stages in order:
#   extract -> grid -> crossdomain -> aggregate -> speed
# Any single stage name runs just that stage. --only FAMILY is forwarded to the
# stage(s). DRY=1 is honored end-to-end (each stage echoes 'DRYCMD: ...').
#
# Examples:
#   bash scripts/maritime/run.sh all
#   bash scripts/maritime/run.sh grid --only infiray
#   DRY=1 bash scripts/maritime/run.sh all | grep '^DRYCMD: '

set -uo pipefail
HERE="$(dirname "$(readlink -f "$0")")"

STAGE=${1:-all}
[ $# -gt 0 ] && shift   # remaining args (e.g. --only FAMILY) forwarded to stages

run_stage() {  # stage-name [args...]
  echo "==================== STAGE: $1 ===================="
  bash "$HERE/$1.sh" "${@:2}"
}

case "$STAGE" in
  all)
    run_stage extract "$@"
    run_stage grid "$@"
    run_stage crossdomain "$@"
    run_stage aggregate "$@"
    run_stage speed "$@"
    ;;
  extract|grid|crossdomain|aggregate|speed)
    run_stage "$STAGE" "$@"
    ;;
  *)
    echo "run.sh: unknown stage '$STAGE' (all|extract|grid|crossdomain|aggregate|speed)" >&2
    exit 2
    ;;
esac
