#!/usr/bin/env bash
set -euo pipefail

ROOT=${AGILLM4_DISAGG_ROOT:-/root/agillm4_disagg}
mkdir -p "$ROOT"
now=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [ -f "$ROOT/.drain_flush" ]; then
  label=$(cat "$ROOT/.drain_flush" 2>/dev/null || echo manual)
  rm -f "$ROOT/.drain_flush"
  printf '{"event":"agillm4_boundary","action":"flush","label":"%s","time":"%s"}\n' "$label" "$now" | tee "$ROOT/.last_boundary_action"
  exit 10
fi

if [ -f "$ROOT/.drain_stop" ]; then
  printf '{"event":"agillm4_boundary","action":"stop","time":"%s"}\n' "$now" | tee "$ROOT/.last_boundary_action"
  exit 20
fi

printf '{"event":"agillm4_boundary","action":"continue","time":"%s"}\n' "$now" > "$ROOT/.last_boundary_action"
exit 0
