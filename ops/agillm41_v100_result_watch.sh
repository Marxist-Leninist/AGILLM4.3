#!/usr/bin/env bash
LOG=/workspace/agillm41_master_train.log; OUT=/workspace/agillm41_v100_gpu_result.json
while true; do
  line=$(grep 'async_side_update_applied' "$LOG" 2>/dev/null | grep vast-v100 | tail -1)
  tok=$(echo "$line" | grep -oE '"tokens": [0-9]+' | grep -oE '[0-9]+')
  tps=$(echo "$line" | grep -oE '"tok_per_sec": [0-9.]+' | grep -oE '[0-9.]+')
  if [ -n "$tok" ] && [ "$tok" != "131072" ]; then
    echo "{\"tokens\":$tok,\"tok_per_sec\":$tps,\"at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"note\":\"V100 new-dims (batch/block>1) result\"}" > "$OUT"
    echo "CAPTURED: tokens=$tok tok/s=$tps"; exit 0
  fi
  sleep 30
done
