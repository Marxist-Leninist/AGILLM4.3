#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-tok-s", type=float, default=3357.167)
    args = ap.parse_args()
    rows = []
    for item in args.updates:
        upd = torch.load(item, map_location="cpu", weights_only=False)
        rows.append({k: upd.get(k) for k in (
            "worker_id", "host", "block_id", "layers", "steps", "batch_size",
            "block_size", "tokens", "wall_sec", "tok_per_sec", "losses"
        )})
    wall = max((float(r["wall_sec"] or 0.0) for r in rows), default=0.0)
    accepted_tokens = sum(int(r["tokens"] or 0) for r in rows)
    by_block = {}
    for r in rows:
        by_block[int(r["block_id"])] = max(by_block.get(int(r["block_id"]), 0), int(r["tokens"] or 0))
    unique_tokens = sum(by_block.values())
    result = {
        "event": "agillm4_distributed_training_benchmark",
        "baseline_4090_tok_per_sec": float(args.baseline_tok_s),
        "workers": rows,
        "accepted_updates": len(rows),
        "accepted_tokens": accepted_tokens,
        "round_wall_sec": wall,
        "master_tok_per_sec": accepted_tokens / max(wall, 1e-9),
        "unique_block_tokens": unique_tokens,
        "unique_block_tok_per_sec": unique_tokens / max(wall, 1e-9),
        "worker_tok_per_sec_sum": sum(float(r["tok_per_sec"] or 0.0) for r in rows),
        "speed_ratio_unique_vs_4090": (unique_tokens / max(wall, 1e-9)) / max(float(args.baseline_tok_s), 1e-9),
        "note": "Non-destructive AGILLM4 benchmark. Workers use real AGILLM4 runtime blocks/MoE/V4-Pro vocab; shared heads/embedding/norm are frozen and only block slices are optimized.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
