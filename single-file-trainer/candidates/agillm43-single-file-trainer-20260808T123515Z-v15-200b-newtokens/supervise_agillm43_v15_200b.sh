#!/usr/bin/env bash
set -euo pipefail

LAUNCHER=/workspace/agillm43_v15_release/launch_agillm43_v15_200b.sh
LAUNCHER_SHA=ac83e7c073e15d98fb4dcc44471ca5ffd93cfe9f7379ee1c6e938f7ae2768c12
SAVE_DIR=/workspace/agillm43_production_v15_200b
STATE=${SAVE_DIR}/run_state.json
LATEST=${SAVE_DIR}/training_latest.json
LOG=${SAVE_DIR}/supervisor.log
SUPERVISOR_LOCK=${SAVE_DIR}/supervisor.lock
STALE_SEC=1800
FLUSH_GRACE_SEC=600
TERM_GRACE_SEC=300
MAX_CONSECUTIVE_FAILURES=3

mkdir -p "${SAVE_DIR}"
exec 8>"${SUPERVISOR_LOCK}"
if ! flock -n 8; then
  echo "the v15 supervisor is already running" >&2
  exit 3
fi
if [[ ! ${LAUNCHER_SHA} =~ ^[0-9a-f]{64}$ ]]; then
  echo "supervisor release is not hash-bound" >&2
  exit 2
fi
printf '%s  %s\n' "${LAUNCHER_SHA}" "${LAUNCHER}" | sha256sum --check --status

budget_complete() {
  [[ -f ${LATEST} ]] || return 1
  /opt/conda/bin/python3 - "${LATEST}" <<'PY' >/dev/null 2>&1
import hashlib, json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not isinstance(data, dict):
    raise SystemExit(1)
claimed = data.get("record_sha256")
body = dict(data); body.pop("record_sha256", None)
actual = hashlib.sha256(json.dumps(
    body, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()
if claimed != actual:
    raise SystemExit(1)
budget = data.get("continuation_budget")
if not isinstance(budget, dict):
    raise SystemExit(1)
budget_claimed = budget.get("sha256")
budget_body = dict(budget); budget_body.pop("sha256", None)
budget_actual = hashlib.sha256(json.dumps(
    budget_body, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()
checkpoint = data.get("checkpoint")
if (budget_claimed != budget_actual
        or data.get("schema") != "agillm43.production.pointer.v1"
        or data.get("role") != "training"
        or data.get("production_schema") != "agillm43.production.v15"
        or budget.get("target_new_tokens") != 200_000_000_000
        or budget.get("anchor_checkpoint_sha256") != "ea90d4f2ddfa1b81e7ab782df02b4f023107f7595b26dfee82818f4c225cb1d1"
        or budget.get("anchor_step") != 1_729_310
        or budget.get("anchor_seen_tok") != 84_999_045_120
        or budget.get("nominal_tokens_per_commit") != 49_152
        or budget.get("full_commits") != 4_069_010
        or budget.get("final_commit_tokens") != 20_480
        or budget.get("target_commits") != 4_069_011
        or budget.get("commits_since_anchor") != 4_069_011
        or budget.get("new_tokens_committed") != 200_000_000_000
        or budget.get("remaining_new_tokens") != 0
        or budget.get("quality_gate_mode") != "telemetry_only"
        or budget.get("quality_gate_can_block") is not False
        or data.get("quality_gate_mode") != "telemetry_only"
        or data.get("quality_gate_can_block") is not False
        or not isinstance(checkpoint, dict)):
    raise SystemExit(1)
path = pathlib.Path(str(checkpoint.get("path") or ""))
sidecar = path.with_suffix(".sha256")
if not path.is_file() or not sidecar.is_file():
    raise SystemExit(1)
parts = sidecar.read_text().split()
expected = str(checkpoint.get("sha256") or "")
if (len(parts) != 2 or parts[0].lower() != expected
        or pathlib.Path(parts[1]).name != path.name
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
    raise SystemExit(1)
package = checkpoint.get("package")
if not isinstance(package, dict):
    raise SystemExit(1)
package_claimed = package.get("sha256")
package_body = dict(package); package_body.pop("sha256", None)
package_actual = hashlib.sha256(json.dumps(
    package_body, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()
shard_name = package.get("shard_dir")
entries = package.get("entries")
if (package_claimed != package_actual or not isinstance(shard_name, str)
        or pathlib.Path(shard_name).name != shard_name
        or not isinstance(entries, list) or not entries):
    raise SystemExit(1)
shard_dir = path.parent / shard_name
for entry in entries:
    shard = shard_dir / str(entry.get("shard") or "")
    if (shard.parent != shard_dir or not shard.is_file()
            or shard.stat().st_size != entry.get("nbytes")):
        raise SystemExit(1)
    digest = hashlib.sha256()
    with shard.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry.get("sha256"):
        raise SystemExit(1)
tokenizer = data.get("tokenizer")
if not isinstance(tokenizer, dict):
    raise SystemExit(1)
tokenizer_path = pathlib.Path(str(tokenizer.get("path") or ""))
tokenizer_sha = str(tokenizer.get("sha256") or "")
if (not tokenizer_path.is_file()
        or tokenizer_path.stat().st_size != tokenizer.get("nbytes")
        or str(tokenizer.get("checkpoint_path") or "") != str(path)
        or str(tokenizer.get("backend_sha256") or "") != "0e546c51529290ed4b18dc8b3e481a9ea39a2df2175b65ecb281af6bac79af73"
        or hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != tokenizer_sha):
    raise SystemExit(1)
expected_public = dict(data)
expected_public.pop("record_sha256", None)
expected_public["role"] = "public"
expected_public["record_sha256"] = hashlib.sha256(json.dumps(
    expected_public, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()
public_path = pathlib.Path(sys.argv[1]).parent / "latest.json"
try:
    public = json.loads(public_path.read_text())
except Exception:
    raise SystemExit(1)
if public != expected_public:
    # The launcher owns crash reconciliation. Returning incomplete here makes
    # the supervisor invoke it instead of silently exiting with stale public state.
    raise SystemExit(1)
raise SystemExit(0)
PY
}

mtime_or_zero() {
  if [[ -e $1 ]]; then stat -c %Y "$1" 2>/dev/null || echo 0; else echo 0; fi
}

budget_progress() {
  if [[ ! -f ${STATE} ]]; then echo -1; return; fi
  /opt/conda/bin/python3 - "${STATE}" "$1" <<'PY' 2>/dev/null || echo -1
import hashlib, json, pathlib, sys
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text())
    budget = data.get("continuation_budget")
    expected_pid = int(sys.argv[2])
    if (data.get("schema") != "agillm.run_state.v1"
            or data.get("production_schema") != "agillm43.production.v15"
            or data.get("pid") != expected_pid
            or not isinstance(budget, dict)):
        print(-1); raise SystemExit(0)
    claimed = budget.get("sha256")
    body = dict(budget); body.pop("sha256", None)
    actual = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if (claimed != actual
            or budget.get("target_new_tokens") != 200_000_000_000
            or budget.get("anchor_checkpoint_sha256") != "ea90d4f2ddfa1b81e7ab782df02b4f023107f7595b26dfee82818f4c225cb1d1"):
        print(-1); raise SystemExit(0)
    value = budget.get("new_tokens_committed") if isinstance(budget, dict) else -1
    print(value if type(value) is int and value >= 0 else -1)
except Exception:
    print(-1)
PY
}

find_live_trainer() {
  local -a matches=()
  mapfile -t matches < <(
    pgrep -f '[p]ython(3)? .*agillm41_repair_v5_atomic_v15.py train' || true
  )
  if (( ${#matches[@]} > 1 )); then
    printf 'multiple v15 trainer processes are live: %s\n' "${matches[*]}" >&2
    return 2
  fi
  if (( ${#matches[@]} == 1 )); then
    printf '%s\n' "${matches[0]}"
    return 0
  fi
  return 1
}

durable_progress() {
  if [[ ! -f ${LATEST} ]]; then echo -1; return; fi
  /opt/conda/bin/python3 - "${LATEST}" <<'PY' 2>/dev/null || echo -1
import json, pathlib, sys
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text())
    budget = data.get("continuation_budget")
    value = budget.get("new_tokens_committed") if isinstance(budget, dict) else -1
    print(value if type(value) is int and value >= 0 else -1)
except Exception:
    print(-1)
PY
}

failures=0
while ! budget_complete; do
  adopted=0
  set +e
  trainer_pid=$(find_live_trainer)
  find_rc=$?
  set -e
  if (( find_rc == 0 )); then
    adopted=1
    printf '[%(%FT%TZ)T] adopting live trainer pid=%s\n' \
      -1 "${trainer_pid}" >>"${LOG}"
  elif (( find_rc == 1 )); then
    printf '[%(%FT%TZ)T] starting launcher\n' -1 >>"${LOG}"
    "${LAUNCHER}" 8>&- >>"${LOG}" 2>&1 &
    trainer_pid=$!
  else
    printf '[%(%FT%TZ)T] refusing ambiguous trainer adoption\n' -1 >>"${LOG}"
    exit 2
  fi
  launched_at=$(date +%s)
  last_progress_at=${launched_at}
  last_progress=$(durable_progress)
  stale_action_at=0
  baseline_latest=$(mtime_or_zero "${LATEST}")
  run_progressed=0

  while kill -0 "${trainer_pid}" 2>/dev/null; do
    sleep 60
    now=$(date +%s)
    current_progress=$(budget_progress "${trainer_pid}")
    if (( current_progress >= 0 && current_progress < last_progress )); then
      last_progress=${current_progress}
      last_progress_at=${now}
      stale_action_at=0
    fi
    if (( current_progress > last_progress )); then
      last_progress=${current_progress}
      last_progress_at=${now}
      stale_action_at=0
      run_progressed=1
    fi
    age=$(( now - last_progress_at ))
    if (( age < STALE_SEC )); then
      continue
    fi

    if (( stale_action_at == 0 )); then
      stale_action_at=${now}
      printf '[%(%FT%TZ)T] stale progress (%ss); requesting checkpoint flush pid=%s\n' \
        -1 "${age}" "${trainer_pid}" >>"${LOG}"
      touch "${SAVE_DIR}/FLUSH_NOW"
      kill -USR1 "${trainer_pid}" 2>/dev/null || true
      continue
    fi

    latest_mtime=$(mtime_or_zero "${LATEST}")
    if (( latest_mtime > baseline_latest )); then
      baseline_latest=${latest_mtime}
      printf '[%(%FT%TZ)T] checkpoint flush completed without token progress; grace continues\n' \
        -1 >>"${LOG}"
    fi
    if (( now - stale_action_at >= FLUSH_GRACE_SEC )); then
      printf '[%(%FT%TZ)T] flush grace expired; requesting graceful termination pid=%s\n' \
        -1 "${trainer_pid}" >>"${LOG}"
      kill -TERM "${trainer_pid}" 2>/dev/null || true
      term_started=${now}
      while kill -0 "${trainer_pid}" 2>/dev/null \
          && (( $(date +%s) - term_started < TERM_GRACE_SEC )); do
        sleep 15
      done
      if kill -0 "${trainer_pid}" 2>/dev/null; then
        printf '[%(%FT%TZ)T] trainer remained hung; killing process only, preserving checkpoints\n' \
          -1 >>"${LOG}"
        kill -KILL "${trainer_pid}" 2>/dev/null || true
      fi
      break
    fi
  done

  if (( adopted == 0 )); then
    set +e
    wait "${trainer_pid}"
    rc=$?
    set -e
  else
    rc=0
  fi
  if budget_complete; then
    printf '[%(%FT%TZ)T] exact 200B continuation complete\n' -1 >>"${LOG}"
    exit 0
  fi
  if (( run_progressed == 1 || $(date +%s) - launched_at > 3600 )); then
    failures=0
  else
    failures=$(( failures + 1 ))
  fi
  printf '[%(%FT%TZ)T] trainer exit rc=%s consecutive_failures=%s\n' \
    -1 "${rc}" "${failures}" >>"${LOG}"
  if (( failures >= MAX_CONSECUTIVE_FAILURES )); then
    printf '[%(%FT%TZ)T] stopping after repeated failures; integrity state is preserved\n' \
      -1 >>"${LOG}"
    exit 1
  fi
  sleep 60
done

printf '[%(%FT%TZ)T] exact 200B continuation already complete\n' -1 >>"${LOG}"
