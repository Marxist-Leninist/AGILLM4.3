# AGILLM4.3 DBlock Coverage Guard 20260614T041641Z

The master trainer now keeps DBlock VRAM savings while preventing layer-block starvation.

- Loss-balanced scheduling still prioritizes blocks with higher EMA loss.
- Hard stale guard: force the stalest block after 64 unselected DBlock steps.
- Hard count-skew guard: force the least-trained block when max/min block counts exceed 1.35x.
- Soft bonuses nudge stale and under-sampled blocks before hard guards trigger.
- DBlock diagnostics now include stale counters: `stale=[...]`.

Live hotpatch restarted from `pretrain_step01735998.pt` and printed:
`[dblock] coverage_guard explore=0.080 max_skew=1.35 max_stale_steps=64`.
