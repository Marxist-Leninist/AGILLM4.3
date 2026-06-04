#!/usr/bin/env bash
set -Eeuo pipefail

MAINLINE="${AGILLM41_MAINLINE:-/workspace/agillm41-mainline}"
SAVE_DIR="${AGILLM41_SAVE_DIR:-/workspace/agillm4_4090_ckpts}"
ROUND_ROOT="${AGILLM41_ROUND_ROOT:-/workspace/agillm41_side_rounds}"
GETH_HOST="${AGILLM41_GETH_HOST:-root@5.75.217.57}"
GETH_KEY="${AGILLM41_GETH_KEY:-/root/.ssh/agillm41_geth_ed25519}"
GETH_WORKER_ROOT="${AGILLM41_GETH_WORKER_ROOT:-/root/agillm41_worker}"
OPPORTUNISTIC_ROOT="${AGILLM41_OPPORTUNISTIC_ROOT:-/root/agillm41_opportunistic}"
INTERVAL_SEC="${AGILLM41_SIDE_CYCLE_SEC:-3600}"
THREADS="${AGILLM41_SIDE_THREADS:-8}"
SMALL_NODE_THREADS="${AGILLM41_SMALL_NODE_THREADS:-2}"
WORKERS_SPEC="${AGILLM41_WORKERS:-geth:0,mcp:1,prime:2,communist-web:3,laptop-auto:0}"
KEEP_ROUNDS="${AGILLM41_SIDE_KEEP_ROUNDS:-2}"

latest_ckpt() {
  python - "$SAVE_DIR" <<'PY'
import json, sys
from pathlib import Path
save = Path(sys.argv[1])
latest = save / "latest.json"
if latest.exists():
    try:
        path = Path(json.loads(latest.read_text()).get("path", ""))
        if path.exists() and path.stat().st_size > 0:
            print(path)
            raise SystemExit
    except Exception:
        pass
matches = sorted(save.glob("pretrain_step*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
if not matches:
    raise SystemExit("no checkpoint found")
print(matches[0])
PY
}

copy_to_geth() {
  local out_dir="$1" base="$2"
  ssh -i "$GETH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$GETH_HOST" \
    "mkdir -p '$GETH_WORKER_ROOT/packages/$base'"
  scp -i "$GETH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no \
    "$out_dir/shared_frozen.pt" \
    "$out_dir/manifest.json" \
    "$out_dir"/lease_*_block*_agillm4bench.pt \
    "$GETH_HOST:$GETH_WORKER_ROOT/packages/$base/"
}

prune_generated_artifacts() {
  find "$ROUND_ROOT" -maxdepth 1 -type d -name 'side_cycle_*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | awk -v keep="$KEEP_ROUNDS" 'NR>keep {sub(/^[^ ]+ /,""); print}' \
    | xargs -r rm -rf
  ssh -i "$GETH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$GETH_HOST" \
    "find '$GETH_WORKER_ROOT/packages' -maxdepth 1 -type d -name 'side_cycle_*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk -v keep='$KEEP_ROUNDS' 'NR>keep {sub(/^[^ ]+ /,\"\"); print}' | xargs -r rm -rf; find '$GETH_WORKER_ROOT/updates' -maxdepth 1 -type f -name 'side_cycle_*_update.pt' -mmin +120 -delete; find '$OPPORTUNISTIC_ROOT/updates' -maxdepth 1 -type f -name 'laptop-auto_*.pt' -mmin +240 -delete"
  ssh -i "$GETH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$GETH_HOST" \
    "for h in 10.0.1.20 10.0.1.30 10.0.1.1; do ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 \"\$h\" \"find '$GETH_WORKER_ROOT/packages' -maxdepth 1 -type d -name 'side_cycle_*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk -v keep='$KEEP_ROUNDS' 'NR>keep {sub(/^[^ ]+ /,\\\"\\\"); print}' | xargs -r rm -rf; find '$GETH_WORKER_ROOT/updates' -maxdepth 1 -type f -name 'side_cycle_*_update.pt' -mmin +120 -delete\" || true; done"
}

cycle_once() {
  local ckpt stamp base out_dir
  ckpt="$(latest_ckpt)"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  base="side_cycle_${stamp}"
  out_dir="$ROUND_ROOT/$base"
  mkdir -p "$out_dir"
  cd "$MAINLINE"
  python agillm4/training_bench/agillm4_export_bench_packages.py \
    --ckpt "$ckpt" \
    --out-dir "$out_dir" \
    --workers "$WORKERS_SPEC" \
    --steps 1 --batch-size 1 --block-size 128 \
    --runtime agillm41.py --source __default__ \
    --attn-backend sublinear \
    --sublinear-window 128 --sublinear-stride 128 --sublinear-max-anchors 128 --sublinear-chunk 128 \
    --sublinear-sinks 4 --sublinear-recent-anchors 64 \
    --objective-mode stochastic --ar-prob 0.70 --sat-prob 0.15 --nat-prob 0.15 \
    --ar-loss-tokens 64 --sat-loss-tokens 0 --nat-loss-tokens 64 \
    --nat-mask-ratio 0.5 --nat-max-tokens 128
  copy_to_geth "$out_dir" "$base"
  ssh -i "$GETH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$GETH_HOST" \
    "cd '$GETH_WORKER_ROOT' && AGILLM41_SIDE_THREADS='$THREADS' AGILLM41_SMALL_NODE_THREADS='$SMALL_NODE_THREADS' bash ./agillm41_dispatch_side_round.sh '$base' && /root/agillm3_geth_cpu/venv/bin/python code/agillm4_publish_opportunistic_lease.py --export-dir '$GETH_WORKER_ROOT/packages/$base' --root '$OPPORTUNISTIC_ROOT' --worker-id laptop-auto --source-worker laptop-auto"
  printf '{"event":"side_cycle_published","base":"%s","ckpt":"%s","at":"%s"}\n' "$base" "$ckpt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  prune_generated_artifacts || true
}

if [ "${1:-}" = "--once" ]; then
  cycle_once
  exit 0
fi

while true; do
  cycle_once || true
  sleep "$INTERVAL_SEC"
done
