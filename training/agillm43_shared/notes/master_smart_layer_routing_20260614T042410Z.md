# AGILLM4.3 Master Smart Layer Routing 20260614T042410Z

The master trainer now uses 4 smart-routed DBlock layer bands instead of 2 macro halves:

- Block 0: layers 0-6
- Block 1: layers 7-13
- Block 2: layers 14-20
- Block 3: layers 21-27

The scheduler remains loss-balanced, with coverage guards:

- exploration: 0.080
- max count skew: 1.35
- max stale steps: 64
- stale/undertrained soft bonuses

Live restart:

- Restart checkpoint: `pretrain_step01736538.pt`
- Launch: `--dblock_blocks 4 --batch_size 18 --block 1536 --attn_backend sdpa --optimizer adamw8bit`
- First post-restart DBlock diagnostics covered all layers 0-27 with no missing layers.

This is master-side routing. Federation leases continue to route finer layer slices and master applies them through async side updates.
