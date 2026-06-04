#!/usr/bin/env bash
set -euo pipefail

ROOT=${AGILLM4_DISAGG_ROOT:-/root/agillm4_disagg}
cmd=${1:-status}
label=${2:-manual}

case "$cmd" in
  flush)
    mkdir -p "$ROOT"
    printf '%s' "$label" > "$ROOT/.drain_flush"
    rm -f "$ROOT/.drain_stop"
    echo "requested AGILLM4 clean round-boundary flush label=$label"
    ;;
  stop)
    mkdir -p "$ROOT"
    : > "$ROOT/.drain_stop"
    echo "requested AGILLM4 clean round-boundary stop"
    ;;
  clear)
    rm -f "$ROOT/.drain_flush" "$ROOT/.drain_stop" "$ROOT/.last_boundary_action"
    echo "cleared AGILLM4 boundary markers"
    ;;
  status)
    echo "root=$ROOT"
    echo "round=$(cat "$ROOT/round.txt" 2>/dev/null || echo unknown)"
    [ -f "$ROOT/.drain_flush" ] && echo "drain_flush=$(cat "$ROOT/.drain_flush")"
    [ -f "$ROOT/.drain_stop" ] && echo "drain_stop=1"
    [ -f "$ROOT/.last_boundary_action" ] && cat "$ROOT/.last_boundary_action"
    pgrep -af "agillm4.*disagg\\|run_agillm4" || true
    ;;
  *)
    echo "usage: $0 {status|flush [label]|stop|clear}" >&2
    exit 2
    ;;
esac
