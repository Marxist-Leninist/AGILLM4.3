#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${1:?usage: agillm41_dispatch_side_round.sh package_dir_basename}"
ROOT="${AGILLM41_WORKER_ROOT:-/root/agillm41_worker}"
PYTHON="${AGILLM41_PYTHON:-/root/agillm3_geth_cpu/venv/bin/python}"
REMOTE_PYTHON="${AGILLM41_REMOTE_PYTHON:-/root/agillm35_worker/venv/bin/python}"
GETH_THREADS="${AGILLM41_SIDE_THREADS:-8}"
SMALL_THREADS="${AGILLM41_SMALL_NODE_THREADS:-2}"

mkdir -p "$ROOT/logs" "$ROOT/updates"

run_local_geth() {
  AGILLM41_SIDE_THREADS="$GETH_THREADS" bash "$ROOT/run_geth_agillm41_side_once.sh" "$BASE"
}

remote_host() {
  case "$1" in
    mcp) printf '10.0.1.20' ;;
    prime) printf '10.0.1.30' ;;
    communist-web) printf '10.0.1.1' ;;
    *) return 1 ;;
  esac
}

remote_block() {
  case "$1" in
    mcp) printf '1' ;;
    prime) printf '2' ;;
    communist-web) printf '3' ;;
    *) return 1 ;;
  esac
}

stage_remote_code() {
  local worker="$1" host="$2"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "mkdir -p '$ROOT/code' '$ROOT/packages/$BASE' '$ROOT/updates' '$ROOT/logs'"
  scp -o BatchMode=yes -o StrictHostKeyChecking=no \
    "$ROOT/code/agillm41.py" \
    "$ROOT/code/agillm4_slice_bench_worker.py" \
    "$host:$ROOT/code/"
}

run_remote_worker() {
  local worker="$1" host block pkg out log
  host="$(remote_host "$worker")"
  block="$(remote_block "$worker")"
  pkg="lease_${worker}_block${block}_agillm4bench.pt"
  out="$ROOT/updates/${BASE}_${worker}_b${block}_update.pt"
  log="$ROOT/logs/${BASE}_${worker}_b${block}.relay.log"
  rm -f "$out" "$out.tmp" "$log"
  (
    stage_remote_code "$worker" "$host"
    scp -o BatchMode=yes -o StrictHostKeyChecking=no \
      "$ROOT/packages/$BASE/shared_frozen.pt" \
      "$ROOT/packages/$BASE/$pkg" \
      "$host:$ROOT/packages/$BASE/"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" \
      "cd '$ROOT' && export OMP_NUM_THREADS='$SMALL_THREADS' MKL_NUM_THREADS='$SMALL_THREADS' OPENBLAS_NUM_THREADS='$SMALL_THREADS' PYTHONWARNINGS='ignore::FutureWarning'; '$REMOTE_PYTHON' -u code/agillm4_slice_bench_worker.py --package 'packages/$BASE/$pkg' --shared 'packages/$BASE/shared_frozen.pt' --runtime code/agillm41.py --out '$out' --device cpu --threads '$SMALL_THREADS'"
    scp -o BatchMode=yes -o StrictHostKeyChecking=no "$host:$out" "$out.tmp"
    mv "$out.tmp" "$out"
    printf '{"event":"remote_side_update_returned","worker":"%s","host":"%s","base":"%s","out":"%s","at":"%s"}\n' "$worker" "$host" "$BASE" "$out" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ) > "$log" 2>&1 &
  printf 'started remote worker=%s host=%s block=%s pid=%s log=%s\n' "$worker" "$host" "$block" "$!" "$log"
}

run_local_geth
run_remote_worker mcp
run_remote_worker prime
run_remote_worker communist-web
