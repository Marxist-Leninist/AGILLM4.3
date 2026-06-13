#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/agillm41_public_join
SPOOL="$ROOT/spool"
PY=/root/agillm41_opportunistic/venv/bin/python
HOST=/root/AGILLM4.1/public_join/agillm41_network_host.py
BASE=https://join.opentransformers.online
PKG_ROOT=/root/agillm41_worker/packages
STATE="$ROOT/last_seeded_cycle"
LOG="$ROOT/logs/seed_latest_leases.log"

mkdir -p "$SPOOL" "$ROOT/logs" /var/www/agillm-pkg
if ! findmnt /var/www/agillm-pkg >/dev/null 2>&1; then
  mount --bind "$PKG_ROOT" /var/www/agillm-pkg
fi
chmod -R a+rX /root/agillm41_worker/packages 2>/dev/null || true  # nginx /pkg/ static readability

# reap expired / abandoned leases (errored or stale) so leased/ does not accumulate
# and a node is not stuck holding a dead lease. Also reap quarantine results > 2h.
_now=$(date -u +%s)
for _lf in "$SPOOL"/leased/*.json; do
  [ -e "$_lf" ] || continue
  _exp=$(python3 -c "import json,sys;print(int(json.load(open(sys.argv[1])).get('expires_at',0)))" "$_lf" 2>/dev/null || echo 0)
  if [ "${_exp:-0}" -lt "$_now" ]; then rm -f "$_lf"; fi
done
find "$SPOOL/quarantine" -type f -mmin +120 -delete 2>/dev/null || true
latest=$(find "$PKG_ROOT" -maxdepth 1 -type d -name 'side_cycle_*' ! -name '*_gpu' | sort | tail -1)
if [[ -z "${latest:-}" || ! -d "$latest" ]]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) no side_cycle package dir" | tee -a "$LOG"
  exit 1
fi
cycle=$(basename "$latest")
last=$(cat "$STATE" 2>/dev/null || true)
# Reseed when: forced, a new cycle appeared, OR the available spool ran dry
# (workers consumed all leases) so volunteers always have work to grab.
avail_count=$(find "$SPOOL/available" -name "*.json" 2>/dev/null | wc -l)
if [[ "$last" == "$cycle" && "${1:-}" != "--force" && "$avail_count" -gt 0 ]]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) already seeded $cycle ($avail_count available)" | tee -a "$LOG"
  exit 0
fi

shared="$latest/shared_frozen.pt"
if [[ ! -s "$shared" ]]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) missing shared_frozen in $cycle" | tee -a "$LOG"
  exit 1
fi

rm -f "$SPOOL"/available/*.json 2>/dev/null || true
count=0
for pkg in "$latest"/lease_*_block*_agillm4bench.pt; do
  [[ -s "$pkg" ]] || continue
  "$PY" "$HOST" add-lease \
    --spool "$SPOOL" \
    --secret-file "$ROOT/lease_secret.txt" \
    --public-base-url "$BASE" \
    --package "$pkg" \
    --frozen "$shared" \
    --ttl-sec 7200 \
    --max-result-bytes 1300000000 \
    --metadata network=agillm41-public --metadata tier=cpu \
    --metadata source_cycle="$cycle" >> "$LOG" 2>&1
  count=$((count + 1))
done


# --- seed a GPU-tier lease (bigger batch/block) for GPU volunteers ---
gpu_latest=$(find "$PKG_ROOT" -maxdepth 1 -type d -name 'side_cycle_*_gpu' | sort | tail -1)
if [[ -n "${gpu_latest:-}" && -d "$gpu_latest" && -s "$gpu_latest/shared_frozen.pt" ]]; then
  for gpkg in "$gpu_latest"/lease_*_block*_agillm4bench.pt; do
    [[ -s "$gpkg" ]] || continue
    "$PY" "$HOST" add-lease --spool "$SPOOL" --secret-file "$ROOT/lease_secret.txt" --public-base-url "$BASE" \
      --package "$gpkg" --frozen "$gpu_latest/shared_frozen.pt" --ttl-sec 7200 --max-result-bytes 1300000000 \
      --metadata network=agillm41-public --metadata tier=gpu --metadata source_cycle="$(basename "$gpu_latest")" >> "$LOG" 2>&1
  done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) seeded gpu-tier lease from $(basename "$gpu_latest")" | tee -a "$LOG"
fi

echo "$cycle" > "$STATE"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) seeded $count leases from $cycle" | tee -a "$LOG"
