# AGILLM4.3 DBlock Learned Router - 20260614T044233Z

The master DBlock scheduler now supports an opt-in tiny transformer router for layer-band selection.

Live launcher setting:

```bash
--dblock_blocks 4 --dblock_schedule loss_balanced --dblock_router transformer --dblock_router_blend 0.35 --dblock_router_ramp_steps 256
```

Behavior:

- The model layers remain partitioned into four bands: 0-6, 7-13, 14-20, and 21-27.
- Hard coverage guards still run first: warmup coverage, stale-block forcing, and max count-skew forcing.
- After warmup, the tiny CPU-side transformer predicts per-band training value from EMA loss, stale age, sample counts, sigma band, and position features.
- The learned score ramps in gradually and blends with the existing heuristic instead of replacing it outright.
- The router is runtime-only and does not change the AGILLM checkpoint format.

Validation performed:

- `python3 -m py_compile agillm41.py`
- In-process smoke test of `_dblock_init`, `_choose_block`, and `_update_stats` with the transformer router enabled.
- Live hotpatch restarted the 4090 trainer from checkpoint `pretrain_step01738407.pt` with `--dblock_router transformer`.
