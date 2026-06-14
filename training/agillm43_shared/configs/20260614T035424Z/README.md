# AGILLM4.3 Training Config Snapshot 20260614T035424Z

This snapshot records the live Vast RTX 4090 training configuration in a scrubbed, agent-friendly form.

- Model parameters: 1,221,580,802 total parameters
- Active node: single Vast RTX 4090
- Batch/context: B=10 L=1536
- DBlock lane: 2 blocks over 28 layers
- Attention backend: sdpa
- Optimizer: adamw8bit
- Gradient checkpointing: False
- Latest local checkpoint: /workspace/agillm4_4090_ckpts_active/pretrain_step01732387.pt
- Checkpoint upload repo/prefix: OpenTransformer/AGILLM-4.3 / training/agillm43_shared

## Checkpoint Upload Assessment

At snapshot time the upload loop parent was alive, but the Python upload child had been stuck since the 2026-06-13T08:01Z tick. The latest recorded full checkpoint upload in state was older than the current local checkpoint, so the uploader needed a timeout guard/restart before fresh checkpoint uploads could be trusted.

## Files

- `agillm43_current_vast_config.json`: full scrubbed operational snapshot
- `agillm43_public_config_summary.json`: public-safe summary
- `launch_args.txt`: exact launch argv with secrets scrubbed
