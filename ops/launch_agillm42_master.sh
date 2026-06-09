#!/usr/bin/env bash
# AGILLM-4.2: same arch (Transformer + MoE + DiffusionBlocks + sublinear), tie_kv.
# Reuses the agillm4.1 distributed stack (same save dir + side_updates).
set -Eeuo pipefail
cd /workspace/agillm41-mainline
export TOKENIZERS_PARALLELISM=false
export TOKENIZER_ID=deepseek-ai/DeepSeek-V4-Pro
export AGILLM_ATTN_BACKEND=sublinear
unset PYTORCH_CUDA_ALLOC_CONF
if [ -f /root/.cache/huggingface/token ]; then
  HF_TOKEN="$(tr -d '\r\n' </root/.cache/huggingface/token)"; export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
SAVE_DIR=/workspace/agillm4_4090_ckpts
SIDE_DIR=/workspace/agillm41_side_updates
mkdir -p "$SAVE_DIR" "$SIDE_DIR/incoming" "$SIDE_DIR/accepted" "$SIDE_DIR/rejected"
exec >> /workspace/agillm41_master_train.log 2>&1
echo "LAUNCH_AGILLM42_MASTER (tie_kv) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
SEED_DELTA="$SAVE_DIR/agillm42_tiekv_seed.delta.pt"
RESUME_DELTA="$SAVE_DIR/agillm42_resume.delta.pt"
# Resume from the newest FULL checkpoint, but as a weights-only delta: this resets
# the (8-bit, paged) optimizer and preserves step/seen_tok, which fits in VRAM at B=6.
# A plain --resume reloads full optimizer state and OOMs the 4090.
CONV=0
python3 - "$SAVE_DIR" "$RESUME_DELTA" <<'PY' || CONV=$?
import json, os, sys, glob, torch
d, out = sys.argv[1], sys.argv[2]
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
ck = torch.load(src, map_location="cpu", weights_only=False)
delta = {"delta": True,
         "weights": {k: ck[k] for k in ("core", "ar", "sat", "nat") if k in ck},
         "step": ck.get("step", 0), "seen_tok": ck.get("seen_tok", 0),
         "cfg": ck.get("cfg")}
tmp = out + ".tmp"
torch.save(delta, tmp); os.replace(tmp, out)
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
  --moe_ffn --moe_experts 2 --moe_top_k 1 --moe_mlp_mult 4 --moe_aux_coef 0.01 --moe_z_coef 0.001 \
  --tie_weights --batch_size 6 --block 1024 --amp --attn_backend sublinear \
  --sublinear_window 128 --sublinear_stride 128 --sublinear_max_anchors 128 --sublinear_chunk 128 \
  --sublinear_sinks 4 --sublinear_recent_anchors 64 --no-sublinear_pooled_landmarks \
  --grad_checkpoint --dblock_checkpoint_stride 1 --optimizer paged_adamw8bit --sat_every 4 --nat_every 4 \
  --nat_max_tokens 768 --nat_mask_ratio 0.5 --token_param_ratio 55 \
  --save_dir "$SAVE_DIR" --save_every_sec 3600 --heartbeat_every_sec 300 \
  --empty_cache_every_steps 0 --delta_every_steps 25000 --delta_max_keep 1 --max_ckpts 2 \
  --async_update_dir "$SIDE_DIR/incoming" --async_update_every_steps 100 --async_update_alpha 0.05 \
  --async_update_max_per_check 2 --async_update_max_age_sec 86400 \
  --async_update_accepted_dir "$SIDE_DIR/accepted" --async_update_rejected_dir "$SIDE_DIR/rejected"
