# AGILLM4 Distributed Training Benchmark

This folder contains a non-destructive benchmark harness for the real AGILLM4
DiffusionBlock training path.

It exports copied block-slice leases from a full AGILLM4 checkpoint, runs each
lease on a worker using the live AGILLM4 `Block`, MoE FFN, sublinear attention
mask path, and V4-Pro tokenizer vocab, then collects timing summaries. It does
not modify the source checkpoint.

## Files

- `agillm4_export_bench_packages.py` - export shared frozen tensors and one
  DBlock lease per worker from a full AGILLM4 checkpoint.
- `agillm4_slice_bench_worker.py` - run one worker lease with the live
  `nB300_agillm4.py` runtime plus its companion `dblocks_train.py`,
  `fused_ce.py`, and `anchor_memory.py` sidecars.
- `agillm4_collect_bench_results.py` - collect update `.pt` files into a
  combined throughput summary.

## Example

```bash
python agillm4_export_bench_packages.py \
  --ckpt /workspace/agillm4_4090_ckpts/pretrain_step01317993.pt \
  --out-dir /workspace/agillm4_dist_bench/export_test \
  --workers geth:0,mcp:1,prime:2,communist-web:3 \
  --dblock-blocks 4 \
  --steps 1 \
  --batch-size 1 \
  --block-size 128 \
  --attn-backend sublinear \
  --sublinear-window 128 \
  --sublinear-stride 128 \
  --sublinear-max-anchors 128 \
  --sublinear-chunk 128
```

Each worker then runs its assigned lease:

```bash
python agillm4_slice_bench_worker.py \
  --package lease_geth_block0_agillm4bench.pt \
  --shared shared_frozen.pt \
  --runtime /root/agillm4_worker/runtime/nB300_agillm4.py \
  --out agillm4_bench_update_geth.pt \
  --device cpu \
  --threads 2
```

The first live all-node run on 2026-06-02 used four 7-layer block leases from
`pretrain_step01317993.pt` and completed 512 unique block tokens in 106.724 s,
or 4.797 combined tok/s, versus a 4090 monolithic baseline of 3357.167 tok/s.
This is a correctness/architecture benchmark for CPU worker slices, not a claim
that the CPU mesh is competitive with the 4090 trainer.
