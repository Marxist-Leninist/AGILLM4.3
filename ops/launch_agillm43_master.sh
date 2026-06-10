#!/usr/bin/env bash
# AGILLM-4.3: 4.2 warm-started + shared experts (DeepSeek/ST-MoE style, zero-init output).
# Takes over the production lineage: resumes weights-only from the newest 4.2 full
# checkpoint in the same save dir (missing shared.* keys init fresh = exact 4.2 at
# step 0), sheds optimizer state, then the shared path learns to contribute.
# Reuses the agillm4.1 distributed stack (same save dir + side_updates).
set -Eeuo pipefail
cd /workspace/agillm41-mainline
export TOKENIZERS_PARALLELISM=false
export TOKENIZER_ID=deepseek-ai/DeepSeek-V4-Pro
export AGILLM_ATTN_BACKEND=sublinear
unset PYTORCH_CUDA_ALLOC_CONF
# Use the RAM disk (/dev/shm, tmpfs) for Python temp so imports never fail when the
# 64GB overlay is full -- 'No usable temporary directory' is what crash-looped us.
if [ -d /dev/shm ] && [ -w /dev/shm ]; then
  mkdir -p /dev/shm/agillm_tmp && export TMPDIR=/dev/shm/agillm_tmp TMP=/dev/shm/agillm_tmp TEMP=/dev/shm/agillm_tmp
fi
if [ -f /root/.cache/huggingface/token ]; then
  HF_TOKEN="$(tr -d '\r\n' </root/.cache/huggingface/token)"; export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
SAVE_DIR=/workspace/agillm4_4090_ckpts
SIDE_DIR=/workspace/agillm41_side_updates
mkdir -p "$SAVE_DIR" "$SIDE_DIR/incoming" "$SIDE_DIR/accepted" "$SIDE_DIR/rejected"
exec >> /workspace/agillm41_master_train.log 2>&1
echo "LAUNCH_AGILLM43_MASTER (tie_kv + shared experts) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
SEED_DELTA="$SAVE_DIR/agillm42_tiekv_seed.delta.pt"
RESUME_DELTA="${SHM_DIR:-/dev/shm}/agillm43_resume.delta.pt"; [ -d /dev/shm ] && [ -w /dev/shm ] || RESUME_DELTA="$SAVE_DIR/agillm43_resume.delta.pt"
RESUME_MARK="$(dirname "$RESUME_DELTA")/.agillm43_resume.step"
# Disk hygiene: clear partial saves left by a crashed/OOM-killed write so they
# cannot accumulate and wedge the disk (a full disk -> "No usable temporary
# directory" -> watchdog crash-loop). Keep only the newest 2 full checkpoints.
rm -f "$SAVE_DIR"/*.tmp 2>/dev/null || true
ls -1t "$SAVE_DIR"/pretrain_step*.pt 2>/dev/null | tail -n +3 | xargs -r rm -f 2>/dev/null || true
# Resume from the newest FULL checkpoint, but as a weights-only delta: this resets
# the (8-bit, paged) optimizer and preserves step/seen_tok, which fits in VRAM at B=6.
# A plain --resume reloads full optimizer state and OOMs the 4090. Skip the (4GB)
# rewrite when the resume delta already matches the newest checkpoint step.
CONV=0
python3 - "$SAVE_DIR" "$RESUME_DELTA" "$RESUME_MARK" <<'PY' || CONV=$?
import json, os, sys, glob, re, torch
d, out, mark = sys.argv[1], sys.argv[2], sys.argv[3]
src = ""
try:
    src = json.load(open(os.path.join(d, "latest.json"))).get("path", "")
except Exception:
    src = ""
if not src or not os.path.exists(src):
    c = sorted(glob.glob(os.path.join(d, "pretrain_step*.pt")), key=os.path.getmtime)
    src = c[-1] if c else ""
if not src:
    sys.exit(3)
m = re.search(r"step0*([0-9]+)", os.path.basename(src)); fstep = m.group(1) if m else ""
if fstep and os.path.exists(out) and os.path.exists(mark):
    try:
        if open(mark).read().strip() == fstep:
            print("resume delta already current at step", fstep); sys.exit(0)
    except Exception:
        pass
ck = torch.load(src, map_location="cpu", weights_only=False)
delta = {"delta": True,
         "weights": {k: ck[k] for k in ("core", "ar", "sat", "nat") if k in ck},
         "step": ck.get("step", 0), "seen_tok": ck.get("seen_tok", 0),
         "cfg": ck.get("cfg")}
tmp = out + ".tmp"
torch.save(delta, tmp); os.replace(tmp, out)
if fstep:
    open(mark, "w").write(fstep)
print("converted %s -> resume delta step %s" % (os.path.basename(src), delta["step"]))
PY
if [ "$CONV" -eq 0 ] && [ -f "$RESUME_DELTA" ]; then
  rm -f "$RESUME_DELTA.sha256"
  RESUME_ARG="--resume_delta $RESUME_DELTA"
  echo "RESUME from converted recovery delta: $RESUME_DELTA"
else
  RESUME_ARG="--resume_delta $SEED_DELTA"
  echo "RESUME conversion failed (rc=$CONV); falling back to seed delta: $SEED_DELTA"
fi
exec python -u agillm41.py train --preset agillm4_floor --tie_kv $RESUME_ARG \
  --dblock --dblock_blocks 4 --dblock_schedule loss_balanced --dblock_warmup_steps 16 \
  --dblock_sigma_curriculum_steps 2000 --dblock_log_every 25 --dblock_objective_mode stochastic \
  --dblock_ar_prob 0.70 --dblock_sat_prob 0.15 --dblock_nat_prob 0.15 \
  --dblock_ar_loss_tokens 512 --dblock_sat_loss_tokens 0 --dblock_nat_loss_tokens 512 \
  --moe_ffn --moe_experts 2 --moe_top_k 1 --moe_mlp_mult 4 --moe_shared_experts 1 --moe_shared_mlp_mult 2 --moe_aux_coef 0.01 --moe_z_coef 0.001 \
  --tie_weights --batch_size 6 --block 1024 --amp --attn_backend sublinear \
  --sublinear_window 128 --sublinear_stride 128 --sublinear_max_anchors 128 --sublinear_chunk 128 \
  --sublinear_sinks 4 --sublinear_recent_anchors 64 --no-sublinear_pooled_landmarks \
  --grad_checkpoint --dblock_checkpoint_stride 1 --optimizer paged_adamw8bit --loss_spike_skip 3.0 --sat_every 4 --nat_every 4 \
  --nat_max_tokens 768 --nat_mask_ratio 0.5 --token_param_ratio 55 \
  --val_tokens 32768 --val_every_sec 3600 --data_seed -1 \
  --save_dir "$SAVE_DIR" --save_every_sec 3600 --heartbeat_every_sec 300 \
  --empty_cache_every_steps 0 --delta_every_steps 25000 --delta_max_keep 1 --max_ckpts 2 \
  --async_update_dir "$SIDE_DIR/incoming" --async_update_every_steps 100 --async_update_alpha 0.05 \
  --async_update_max_per_check 2 --async_update_max_age_sec 86400 \
  --async_update_accepted_dir "$SIDE_DIR/accepted" --async_update_rejected_dir "$SIDE_DIR/rejected"
