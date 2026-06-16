# AGILLM4.3 Sublayer Decomposition Probe

Date: 2026-06-16

This is a non-invasive side probe run against a current AGILLM4.3 checkpoint. It does not change the main trainer. The goal is to test whether DiffusionBlocks can be decomposed below whole transformer layers.

Current mainline context while probing:

- Main trainer stayed healthy around B=22, L=1536, about 78k tok/s.
- Runtime is already training 14 DiffusionBlocks of 2 transformer layers each, so the next useful granularity is inside a transformer block.
- Split point tested: attention residual vs FFN/MoE residual.

Probe setup:

- Checkpoint: pretrain_step01961941.pt
- Device: RTX 3090 side process
- Probe dtype: fp32, because fp16 isolated micro-updates produced NaNs.
- Sequence length: 128
- Steps per variant: 10
- Fixed-noise eval before/after used for comparison.

| Layer | Variant | Fixed eval before | Fixed eval after | Eval delta | Peak GB |
|---:|---|---:|---:|---:|---:|
| 0 | full | 12.1092 | 11.7037 | 0.4055 | 1.61 |
| 0 | attention only | 13.0710 | 12.8216 | 0.2494 | 1.19 |
| 0 | FFN/MoE only | 10.2204 | 10.0622 | 0.1582 | 1.53 |
| 0 | attention then FFN alternating | 12.1092 | 11.7105 | 0.3987 | 1.60 |
| 13 | full | 6.6403 | 6.1606 | 0.4797 | 1.61 |
| 13 | attention only | 10.6615 | 10.5777 | 0.0838 | 1.19 |
| 13 | FFN/MoE only | 7.1385 | 6.7966 | 0.3419 | 1.53 |
| 13 | attention then FFN alternating | 6.6403 | 6.2755 | 0.3648 | 1.60 |
| 27 | full | 7.1391 | 6.8540 | 0.2851 | 1.61 |
| 27 | attention only | 11.2548 | 11.2548 | 0.0000 | 1.19 |
| 27 | FFN/MoE only | 7.1391 | 6.8540 | 0.2851 | 1.53 |
| 27 | attention then FFN alternating | 7.1391 | 6.8553 | 0.2838 | 1.60 |

Initial interpretation:

- Sublayer decomposition is worth pursuing.
- Attention-only updates are much weaker by themselves, especially late in the stack.
- FFN/MoE-only updates carry a lot of the useful loss reduction, especially in middle and late layers.
- Alternating attention and FFN microsteps nearly matches full-layer improvement in early and late layers, and is respectable in the middle layer.
- For AGILLM4.3, the likely useful branch is not arbitrary sublayer fragmentation. It is a learned micro-router that chooses among full block, attention-only, FFN/MoE-only, and alternating microsteps based on loss/gradient/VRAM/throughput signals.

Recommended next experiment:

1. Add a gated sublayer-router branch behind flags only.
2. Start with side workers, not master, and never switch mainline until validation loss confirms the benefit.
3. Prioritize FFN/MoE-only and alternating microsteps for cheap side updates.
4. Keep attention-only as an exploratory/repair update, not a default training path.
