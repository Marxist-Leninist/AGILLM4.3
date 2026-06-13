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

## Run Your Own Federated Network

If you want to host your own decentralized training swarm for AGILLM models, you can run the volunteer coordinator and validation endpoints yourself.

1. **Start the Network Host (Coordinator):**
   This script runs a FastHTML web server that distributes training leases to workers and receives asynchronous `.pt` gradient updates.
   ```bash
   python public_join/agillm41_network_host.py \
     --host 0.0.0.0 \
     --port 8787 \
     --spool ./agillm41_lease_spool \
     --public-base-url http://YOUR_IP:8787
   ```

2. **Add Packages (Master Node):**
   Your master training loop exports bench packages which are added to the spool:
   ```bash
   python public_join/agillm41_network_host.py add-lease \
     --spool ./agillm41_lease_spool \
     --package /path/to/exported_bench_pkg.pt \
     --base-ckpt /path/to/base_model.pt
   ```

3. **Validate Results:**
   Once workers submit results to the quarantine directory in the spool, you must validate them before your master applies them.
   ```bash
   python public_join/agillm41_validate_and_credit.py \
     --spool ./agillm41_lease_spool \
     --base-ckpt /path/to/base_model.pt
   ```
   Validated updates will be moved to an `accepted/` directory which your master can then asynchronously merge.
