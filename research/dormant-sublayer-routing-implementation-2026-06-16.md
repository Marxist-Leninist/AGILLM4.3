# Dormant Sublayer Routing Implementation

Date: 2026-06-16

Dormant trainer support was added for future AGILLM4.3 DiffusionBlocks sublayer experiments. The current main run was not restarted and does not pass the new flag, so existing behavior remains unchanged.

New opt-in flag:

```bash
--dblock_sublayer_mode {off,full,attn_only,ffn_only,split_alt,cycle}
```

Default behavior:

- `off` is the default.
- `off` and `full` use normal `Block.forward` behavior.
- Existing training launch profiles do not enable this feature.

Implemented future modes:

- `attn_only`: runs only `LN1 + attention + residual` for the selected DiffusionBlock layer.
- `ffn_only`: runs only `LN2 + FFN/MoE + residual` for the selected DiffusionBlock layer.
- `split_alt`: alternates attention-only and FFN-only by dblock step/block/layer position.
- `cycle`: rotates full, FFN-only, and attention-only by dblock step/block/layer position.

Why this exists:

The 2026-06-16 side probe suggested FFN/MoE-only and attention-then-FFN alternating microsteps can preserve useful learning signal below whole-layer DiffusionBlocks, while attention-only by itself is weak/noisy. This dormant implementation makes it possible to test those paths later without rewriting trainer internals.

Safety notes:

- Do not enable on master without a validation run.
- Start on side workers first.
- The fp16 standalone probe produced NaNs; future mixed-precision runs should use conservative LR, clipping, loss scaling, or fp32 master/update handling.
- Public note intentionally excludes host secrets, tokens, and operational command lines.
