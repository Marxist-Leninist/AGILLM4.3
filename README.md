---
library_name: pytorch
tags:
  - agillm
  - transformer
  - diffusion-block
  - mixture-of-experts
  - single-file
license: other
---

# AGILLM 4.3

AGILLM 4.3 is the AGILLM 4.2 warm start with shared MoE experts and DiffusionBlocks training. The compatibility runtime file is still named `agillm41.py`, but this repo tracks the 4.3 runtime and public volunteer path.

## Public Safety Boundary

This public repo is for inspection, local inference experiments, and untrusted volunteer helpers. It intentionally excludes trusted-core operations, private topology, watchdog launch scripts, live hot configs, SSH paths, API tokens, and checkpoint merge scripts.

Untrusted volunteer nodes should use only the outbound public join flow. The published coordinator is `https://join.opentransformers.online`, and its health endpoint is `https://join.opentransformers.online/health`.

```bash
python public_join/agillm41_join_worker.py \
  --coordinator-url https://join.opentransformers.online \
  --device cpu \
  --threads 2 \
  --loop
```

The worker opens outbound HTTPS only, verifies SHA-256 for lease artifacts, receives short-lived lease tokens only, and submits results to quarantine. Public helper results must be validated before they can affect a checkpoint.

## Files

- `agillm41.py`: latest public AGILLM runtime, including AR/SAT/NAT inference and DiffusionBlocks paths.
- `public_join/`: outbound worker, public lease coordinator, and public validation/points helpers.
- `agillm4/training_bench/agillm4_slice_bench_worker.py`: default public slice worker for leased training packages.
- `distributed_infer/`: public distributed inference harnesses without private launch topology.

## Private Counterpart

Trusted-core operations live in the private repo `Marxist-Leninist/agillm4.3-private` and private HF repo `OpenTransformer/agillm4.3-private`.

## Hugging Face

Public model card and checkpoint lineage: https://hf.co/OpenTransformer/AGILLM-4.3
