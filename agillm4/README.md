# AGILLM4.1 Operations Pack

This pack contains the distributed inference, side-worker, and operations layer
used around the AGILLM4.1 mainline runtime in `agillm41.py`.

## Files

- `distributed_infer/agillm4_distributed_infer.py` - staged AGILLM4.1 AR inference over local or HTTP workers.
- `training_bench/` - non-destructive AGILLM4.1 DiffusionBlock all-node training benchmark.
- `ops/agillm4_make_infer_slim.py` - exports a smaller AR/SAT/NAT inference checkpoint from full or delta checkpoints.
- `ops/agillm4_export_infer_packages.py` - exports coordinator and split-stage packages for staged inference workers.
- `ops/agillm4_boundary_control.sh` - asks a running AGILLM4 loop to flush or stop at a clean round boundary.
- `ops/agillm4_round_boundary_hook.sh` - hook a loop calls after each completed round; exit 10 means save/flush, exit 20 means stop.
- `ops/agillm4_register_opportunistic_pool.sh`, `ops/agillm4_publish_opportunistic_lease.py`,
  `ops/agillm4_laptop_opportunistic_worker.ps1`, `ops/agillm41_vast_side_update_puller.sh`, and
  `ops/agillm41_vast_side_cycle.sh`, and `ops/agillm41_dispatch_side_round.sh` -
  optional AGILLM4.1 laptop side-worker flow. Vast stays master,
  Hetzner stays reliable, and the laptop only contributes async leases when awake.

## Validation Command

Use `compare` before trusting any split:

```bash
python3 /root/agillm4_upgrade/distributed_infer/agillm4_distributed_infer.py compare \
  --agillm4-path /root/agillm35/nB300_agillm4.py \
  --ckpt /path/to/agillm4_infer_slim.pt \
  --prompt "In a practical distributed neural network, the main advantage is" \
  --stage local:0:12 --stage local:12:24 --device cpu
```

The result should have `top1_match: true`; low logit diff means the staged path
is matching monolithic AGILLM4 for that prompt/checkpoint.

## Loop Hook Pattern

Call the hook immediately after a round is merged/exported:

```bash
/root/agillm4_upgrade/ops/agillm4_round_boundary_hook.sh || rc=$?
case "${rc:-0}" in
  0) ;;
  10) save_or_export_master_now ;;
  20) save_or_export_master_now; exit 0 ;;
  *) exit "$rc" ;;
esac
```
