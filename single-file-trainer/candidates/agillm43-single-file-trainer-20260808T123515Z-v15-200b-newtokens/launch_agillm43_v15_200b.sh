#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/conda/bin/python3
TRAINER=/workspace/agillm43_v15_release/agillm41_repair_v5_atomic_v15.py
TRAINER_SHA=efb6847ec10177a3c7c6bedfe531f12dce608252128df64b8ea1c10ec6029216
SEED=/workspace/agillm43_seed_verified/checkpoints/pretrain_step01729310_from01695324_20260808T0011Z.pt
SEED_SHA=ea90d4f2ddfa1b81e7ab782df02b4f023107f7595b26dfee82818f4c225cb1d1
SAVE_DIR=/workspace/agillm43_production_v15_200b
LATEST=${SAVE_DIR}/training_latest.json
LOCK=${SAVE_DIR}/trainer.lock
TOKENIZER_DIR=/workspace/agillm43_repair_stage/v5_release_tokenizer
VAL_FILE=/workspace/agillm43_repair_stage/heldout_v2/token_ids.json

if [[ ! ${TRAINER_SHA} =~ ^[0-9a-f]{64}$ ]]; then
  echo "release is not hash-bound" >&2
  exit 2
fi
printf '%s  %s\n' "${TRAINER_SHA}" "${TRAINER}" | sha256sum --check --status
[[ -x ${PYTHON} ]]

mode=initial
resume=${SEED}
if [[ -e ${LATEST} ]]; then
  selection=$(
    "${PYTHON}" - "${LATEST}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

pointer = pathlib.Path(sys.argv[1])
try:
    data = json.loads(pointer.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"corrupt v15 training pointer: {exc}")
if not isinstance(data, dict):
    raise SystemExit("v15 training pointer is not an object")
claimed_record = data.get("record_sha256")
record_body = dict(data)
record_body.pop("record_sha256", None)
actual_record = hashlib.sha256(json.dumps(
    record_body, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()
if claimed_record != actual_record:
    raise SystemExit("v15 training pointer seal is invalid")
budget = data.get("continuation_budget")
if (data.get("schema") != "agillm43.production.pointer.v1"
        or data.get("role") != "training"
        or data.get("production_schema") != "agillm43.production.v15"):
    raise SystemExit("existing training pointer is not a v15 production pointer")
if not isinstance(budget, dict):
    raise SystemExit("existing v15 pointer has no continuation budget")
claimed_budget = budget.get("sha256")
budget_body = dict(budget)
budget_body.pop("sha256", None)
actual_budget = hashlib.sha256(json.dumps(
    budget_body, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()
if claimed_budget != actual_budget:
    raise SystemExit("existing v15 continuation budget seal is invalid")
if budget.get("target_new_tokens") != 200_000_000_000:
    raise SystemExit("existing v15 pointer has the wrong immutable token budget")
if budget.get("anchor_checkpoint_sha256") != "ea90d4f2ddfa1b81e7ab782df02b4f023107f7595b26dfee82818f4c225cb1d1":
    raise SystemExit("existing v15 pointer has the wrong warm-start anchor")
if (budget.get("anchor_step") != 1_729_310
        or budget.get("anchor_seen_tok") != 84_999_045_120):
    raise SystemExit("existing v15 pointer has the wrong warm-start anchor clocks")
if (budget.get("quality_gate_mode") != "telemetry_only"
        or budget.get("quality_gate_can_block") is not False
        or data.get("quality_gate_mode") != "telemetry_only"
        or data.get("quality_gate_can_block") is not False):
    raise SystemExit("existing v15 pointer permits an output-quality blocker")
integer_fields = (
    "anchor_step", "anchor_seen_tok", "target_new_tokens",
    "new_tokens_committed", "commits_since_anchor", "target_seen_tok",
    "nominal_tokens_per_commit", "full_commits", "final_commit_tokens",
    "target_commits", "remaining_new_tokens",
)
if any(type(budget.get(name)) is not int or budget[name] < 0 for name in integer_fields):
    raise SystemExit("existing v15 pointer has a non-integer budget clock")
if (budget["nominal_tokens_per_commit"] != 49_152
        or budget["full_commits"] != 4_069_010
        or budget["final_commit_tokens"] != 20_480
        or budget["target_commits"] != 4_069_011):
    raise SystemExit("existing v15 pointer has invalid exact-budget arithmetic")
commits = budget["commits_since_anchor"]
expected_new = min(commits, 4_069_010) * 49_152
if commits > 4_069_010:
    expected_new += 20_480
if (commits > 4_069_011
        or budget["new_tokens_committed"] != expected_new
        or budget["remaining_new_tokens"] != 200_000_000_000 - expected_new
        or budget["target_seen_tok"] != budget["anchor_seen_tok"] + 200_000_000_000):
    raise SystemExit("existing v15 pointer budget clocks disagree")
remaining = budget.get("remaining_new_tokens")
if type(remaining) is not int or remaining < 0:
    raise SystemExit("existing v15 pointer has an invalid remaining-token clock")
path = pathlib.Path(str(data.get("path") or ""))
if not path.is_file():
    raise SystemExit(f"v15 child checkpoint is missing: {path}")
checkpoint = data.get("checkpoint")
if not isinstance(checkpoint, dict) or pathlib.Path(str(checkpoint.get("path") or "")) != path:
    raise SystemExit("v15 pointer checkpoint identity is invalid")
checkpoint_sha = str(checkpoint.get("sha256") or "")
if len(checkpoint_sha) != 64:
    raise SystemExit("v15 pointer checkpoint digest is invalid")
sidecar = path.with_suffix(".sha256")
if not sidecar.is_file():
    raise SystemExit(f"v15 child checksum sidecar is missing: {sidecar}")
parts = sidecar.read_text(encoding="utf-8").split()
if len(parts) != 2 or parts[0].lower() != checkpoint_sha or pathlib.Path(parts[1]).name != path.name:
    raise SystemExit("v15 child checksum sidecar binding is invalid")
if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint_sha:
    raise SystemExit("v15 child checkpoint manifest digest mismatch")
package = checkpoint.get("package")
if not isinstance(package, dict):
    raise SystemExit("v15 child checkpoint has no package receipt")
claimed_package = package.get("sha256")
package_body = dict(package)
package_body.pop("sha256", None)
actual_package = hashlib.sha256(json.dumps(
    package_body, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()
if claimed_package != actual_package:
    raise SystemExit("v15 child package receipt seal is invalid")
shard_dir_name = package.get("shard_dir")
if not isinstance(shard_dir_name, str) or pathlib.Path(shard_dir_name).name != shard_dir_name:
    raise SystemExit("v15 child package shard directory is unsafe")
shard_dir = path.parent / shard_dir_name
entries = package.get("entries")
if not shard_dir.is_dir() or not isinstance(entries, list) or not entries:
    raise SystemExit("v15 child package shards are unavailable")
for entry in entries:
    shard = shard_dir / str(entry.get("shard") or "")
    if (shard.parent != shard_dir or not shard.is_file()
            or shard.stat().st_size != entry.get("nbytes")):
        raise SystemExit(f"v15 child package shard is missing or truncated: {shard.name}")
tokenizer = data.get("tokenizer")
if not isinstance(tokenizer, dict):
    raise SystemExit("v15 child pointer has no immutable tokenizer artifact")
tokenizer_path = pathlib.Path(str(tokenizer.get("path") or ""))
tokenizer_sha = str(tokenizer.get("sha256") or "")
if (not tokenizer_path.is_file()
        or tokenizer_path.stat().st_size != tokenizer.get("nbytes")
        or str(tokenizer.get("checkpoint_path") or "") != str(path)
        or str(tokenizer.get("backend_sha256") or "") != "0e546c51529290ed4b18dc8b3e481a9ea39a2df2175b65ecb281af6bac79af73"
        or hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != tokenizer_sha):
    raise SystemExit("v15 child tokenizer artifact integrity check failed")
if remaining == 0:
    for entry in entries:
        shard = shard_dir / entry["shard"]
        digest = hashlib.sha256()
        with shard.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry.get("sha256"):
            raise SystemExit(
                f"completed v15 package shard checksum mismatch: {shard.name}"
            )
    # training_latest is the durable commit point. If power failed between its
    # publication and latest.json, reconstruct the byte-equivalent public role
    # only after the complete package/tokenizer verification above.
    expected_public = dict(data)
    expected_public.pop("record_sha256", None)
    expected_public["role"] = "public"
    expected_public["record_sha256"] = hashlib.sha256(json.dumps(
        expected_public, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    public_pointer = pointer.parent / "latest.json"
    try:
        current_public = json.loads(public_pointer.read_text(encoding="utf-8"))
    except Exception:
        current_public = None
    if current_public != expected_public:
        raw = (json.dumps(
            expected_public, indent=2, sort_keys=True, allow_nan=False
        ) + "\n").encode("utf-8")
        tmp = public_pointer.with_name(
            public_pointer.name + f".reconcile.{os.getpid()}.tmp"
        )
        with tmp.open("xb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, public_pointer)
        dir_fd = os.open(str(public_pointer.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    print("COMPLETED")
    raise SystemExit(0)
print(path)
PY
  )
  if [[ ${selection} == COMPLETED ]]; then
    echo "the exact 200B-new-token continuation is already complete"
    exit 0
  fi
  mode=child
  resume=${selection}
fi

mkdir -p "${SAVE_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "the v15 trainer lock is already held; refusing a duplicate" >&2
  exit 4
fi
if pgrep -af '[p]ython.*agillm41_repair.* train' >/dev/null; then
  echo "a trainer process is already live; refusing a duplicate" >&2
  exit 5
fi
free_kb=$(df -Pk /workspace | awk 'NR==2 {print $4}')
if (( free_kb < 24 * 1024 * 1024 )); then
  echo "less than 24 GiB free on /workspace" >&2
  exit 3
fi
[[ -f ${TOKENIZER_DIR}/tokenizer.json ]]
if [[ ${mode} == initial ]]; then
  [[ -f ${SEED} ]]
  printf '%s  %s\n' "${SEED_SHA}" "${SEED}" | sha256sum --check --status
fi

initial_flags=()
if [[ ${mode} == initial ]]; then
  initial_flags=(
    --reset_optimizer_on_resume
    --migrate_nat_mask_embedding_from_legacy
  )
fi

sources='HuggingFaceFW/fineweb-edu:sample-10BT,HuggingFaceFW/fineweb:CC-MAIN-2024-10,wikimedia/wikipedia:20231101.en,allenai/c4:en,Skylion007/openwebtext,HuggingFaceTB/cosmopedia:web_samples_v2,HuggingFaceTB/smollm-corpus:cosmopedia-v2,EleutherAI/proof-pile-2:all|0.80,allenai/dolma:v1_6-sample|0.70,codeparrot/codeparrot-clean|0.45'

echo "launching AGILLM4.3 v15 mode=${mode} resume=${resume}"
exec env -i \
  AGILLM43_DECOMPRESS_CACHE=0 \
  CUDA_MODULE_LOADING=LAZY \
  CUDA_VISIBLE_DEVICES=0 \
  LC_CTYPE=C.UTF-8 \
  PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  TOKENIZER_DIR="${TOKENIZER_DIR}" \
  TOKENIZER_ID=deepseek-ai/DeepSeek-V4-Pro \
  "${PYTHON}" -B "${TRAINER}" train \
  --preset agillm4_floor \
  --rank 160 \
  --resume "${resume}" \
  --save_dir "${SAVE_DIR}" \
  --additional_tokens 200000000000 \
  --continuation_anchor_sha256 "${SEED_SHA}" \
  --source "${sources}" \
  --block 2048 \
  --batch_size 24 \
  --data_seed 1729310 \
  --optimizer adamw8bit \
  --weight_decay 0.0 \
  --lr_core 3e-6 \
  --lr_head 3e-6 \
  --lr_decay cosine \
  --lr_decay_tokens 200000000000 \
  --lr_schedule_reset_on_resume \
  --lr_warmup_tokens 10000000 \
  --lr_warmup_min_mult 0.10 \
  --lr_min_mult 0.20 \
  --amp \
  --attn_backend sdpa \
  --grad_checkpoint \
  --alibi_mode corrected \
  --alibi_scale 0.0 \
  --tie_weights \
  --tie_kv \
  --moe_ffn \
  --moe_experts 2 \
  --moe_top_k 1 \
  --moe_mlp_mult 4 \
  --moe_shared_experts 1 \
  --moe_shared_mlp_mult 2 \
  --moe_aux_coef 0.01 \
  --moe_z_coef 0.001 \
  --sublinear_window 128 \
  --sublinear_stride 128 \
  --sublinear_max_anchors 128 \
  --sublinear_chunk 128 \
  --sublinear_sinks 4 \
  --sublinear_recent_anchors 64 \
  --no-sublinear_pooled_landmarks \
  --anchor_stride 256 \
  --anchor_max 2048 \
  --anchor_position -1 \
  --dblock \
  --dblock_blocks 14 \
  --dblock_schedule roundrobin \
  --dblock_objective_mode committed_cycle \
  --dblock_router heuristic \
  --dblock_router_blend 0.0 \
  --dblock_router_ramp_steps 64 \
  --dblock_warmup_steps 14 \
  --dblock_log_every 1 \
  --dblock_checkpoint_stride 1 \
  --dblock_checkpoint_skip_tail 0 \
  --dblock_ar_prob 0.80 \
  --dblock_sat_prob 0.10 \
  --dblock_nat_prob 0.10 \
  --dblock_ar_weight 1.0 \
  --dblock_sat_weight 1.0 \
  --dblock_nat_weight 1.0 \
  --dblock_ar_loss_tokens 4096 \
  --dblock_sat_loss_tokens 2048 \
  --dblock_nat_loss_tokens 2048 \
  --dblock_sigma_curriculum_steps 2000 \
  --dblock_sigma_sampling lognormal \
  --dblock_sigma_stratified \
  --dblock_sigma_min 0.002 \
  --dblock_sigma_max 80.0 \
  --dblock_sigma_pmean -1.2 \
  --dblock_sigma_pstd 1.2 \
  --dblock_edm_wmax 5.0 \
  --sat_every 1 \
  --nat_every 1 \
  --nat_loss_weight 1.0 \
  --nat_mask_token_id 2 \
  --nat_mask_ratio 0.5 \
  --nat_span_mask_prob 0.35 \
  --nat_suffix_mask_prob 0.20 \
  --nat_max_tokens 0 \
  --nat_span_max_tokens 0 \
  --dblock_nat_embed_noise_mode off \
  --dblock_nat_embed_noise_scale 1.0 \
  --nat_document_boundary_aware \
  --dblock_fullstack_ar_every 3 \
  --dblock_fullstack_ar_offset 0 \
  --dblock_fullstack_ar_tokens 128 \
  --dblock_fullstack_ar_weight 0.05 \
  --dblock_fullstack_sat_every 3 \
  --dblock_fullstack_sat_offset 1 \
  --dblock_fullstack_sat_tokens 128 \
  --dblock_fullstack_sat_weight 0.10 \
  --dblock_fullstack_nat_every 3 \
  --dblock_fullstack_nat_offset 2 \
  --dblock_fullstack_nat_tokens 128 \
  --dblock_fullstack_nat_weight 0.10 \
  --dblock_fullstack_nat_mask_id -1 \
  --dblock_fullstack_nat_no_valid_crop_limit 8 \
  --dblock_fullstack_anchor_deterministic_eval \
  --loss_spike_skip 3.0 \
  --dblock_fullstack_anchor_spike_skip 3.0 \
  --dblock_fullstack_anchor_max_ce 50.0 \
  --dblock_fullstack_anchor_spike_retry_limit 3 \
  --dblock_local_spike_retry_limit 3 \
  --dblock_local_aux_warn_ce 25.0 \
  --dblock_local_aux_max_ce 50.0 \
  --val_file "${VAL_FILE}" \
  --val_tokens 4096 \
  --val_every_sec 3600 \
  --heartbeat_every_sec 120 \
  --save_every_sec 14400 \
  --max_ckpts 1 \
  --disk_free_floor_gb 24 \
  --delta_every_steps 0 \
  --delta_every_sec 0 \
  --ckpt_codec block-sharded-zstd \
  --no-oom_auto_backoff \
  --repair_fail_fast \
  "${initial_flags[@]}"
