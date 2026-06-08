#!/usr/bin/env python3
"""Export AGILLM4 DBlock benchmark packages from a full checkpoint.

The packages are intentionally non-destructive: workers train a copied slice and
write update/state stats, but the active checkpoint is not modified.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch


def parse_workers(spec: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, block = item.rsplit(":", 1)
        out.append((name.strip(), int(block)))
    return out


def dblock_layers(total_layers: int, blocks: int) -> list[list[int]]:
    span = max(1, total_layers // blocks)
    assign = [list(range(i * span, (i + 1) * span)) for i in range(blocks)]
    assign[-1] = list(range((blocks - 1) * span, total_layers))
    return assign


def local_block_state(core_state: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for local_i, global_i in enumerate(layers):
        src_prefix = f"blocks.{global_i}."
        dst_prefix = f"blocks.{local_i}."
        for key, value in core_state.items():
            if isinstance(key, str) and key.startswith(src_prefix):
                out[dst_prefix + key[len(src_prefix) :]] = value.detach().cpu()
    return out


def token_batches(vocab: int, steps: int, batch_size: int, block_size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    # Keep clear of special tokens; this is a compute benchmark, not a quality run.
    return torch.randint(2, int(vocab), (int(steps), int(batch_size), int(block_size)), generator=gen, dtype=torch.long)


def load_runtime(path: str | Path):
    path = Path(path).resolve()
    os.environ.setdefault("TOKENIZER_ID", "deepseek-ai/DeepSeek-V4-Pro")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("agillm41_export_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import AGILLM4.1 runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agillm41_export_runtime"] = module
    spec.loader.exec_module(module)
    return module


def real_token_batches(runtime: Any, source: str, steps: int, batch_size: int, block_size: int, seed: int) -> torch.Tensor:
    if source == "__default__":
        source = getattr(runtime, "DEFAULT_PRETRAIN_SOURCES")
    total = int(steps) * int(batch_size) * int(block_size)
    stream = runtime.token_stream(source, total, seed=int(seed), streaming=True)
    data = []
    for _ in range(total):
        data.append(int(next(stream)))
    return torch.tensor(data, dtype=torch.long).view(int(steps), int(batch_size), int(block_size))


def main() -> int:
    ap = argparse.ArgumentParser(description="Export AGILLM4 all-node benchmark packages")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", required=True, help="name:block_id comma list")
    ap.add_argument("--dblock-blocks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--max-layers", type=int, default=0, help="export only N layers of the block (rotating window) for low-RAM nodes")
    ap.add_argument("--layer-offset", type=int, default=0, help="rotating start offset within block layer list")
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument("--runtime", default="agillm41.py", help="AGILLM4.1 runtime path used when --source is set")
    ap.add_argument("--source", default="", help="Real token source. Use __default__ for the runtime default pretrain mix; empty keeps synthetic benchmark IDs.")
    ap.add_argument("--attn-backend", choices=["manual", "sdpa", "sublinear"], default="manual")
    ap.add_argument("--sublinear-window", type=int, default=128)
    ap.add_argument("--sublinear-stride", type=int, default=128)
    ap.add_argument("--sublinear-max-anchors", type=int, default=128)
    ap.add_argument("--sublinear-chunk", type=int, default=128)
    ap.add_argument("--sublinear-sinks", type=int, default=4)
    ap.add_argument("--sublinear-recent-anchors", type=int, default=64)
    ap.add_argument("--sublinear-pooled-landmarks", action="store_true")
    ap.add_argument("--objective-mode", choices=["stochastic", "periodic"], default="stochastic")
    ap.add_argument("--ar-prob", type=float, default=0.70)
    ap.add_argument("--sat-prob", type=float, default=0.15)
    ap.add_argument("--nat-prob", type=float, default=0.15)
    ap.add_argument("--ar-loss-tokens", type=int, default=128)
    ap.add_argument("--sat-loss-tokens", type=int, default=0)
    ap.add_argument("--nat-loss-tokens", type=int, default=128)
    ap.add_argument("--nat-mask-ratio", type=float, default=0.5)
    ap.add_argument("--nat-max-tokens", type=int, default=128)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = parse_workers(args.workers)

    start = time.time()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    core = ck["core"]
    vocab = int(core["emb.weight"].shape[0])
    assignments = dblock_layers(int(cfg["layers"]), int(args.dblock_blocks))
    tie_weights = bool(ck.get("tie_weights", False))
    runtime = load_runtime(args.runtime) if args.source else None

    shared = {
        "kind": "agillm4_bench_shared_v1",
        "cfg": cfg,
        "tie_weights": tie_weights,
        "tokenizer_id": ck.get("tokenizer_id"),
        "vocab": vocab,
        "emb_weight": core["emb.weight"].detach().cpu(),
        "ln_weight": core["ln.weight"].detach().cpu(),
        "ln_bias": core["ln.bias"].detach().cpu(),
    }
    if not tie_weights:
        shared["ar"] = {k: v.detach().cpu() for k, v in ck.get("ar", {}).items()}
        shared["sat"] = {k: v.detach().cpu() for k, v in ck.get("sat", {}).items()}
        shared["nat"] = {k: v.detach().cpu() for k, v in ck.get("nat", {}).items()}
    else:
        sat = ck.get("sat", {})
        if "gate.weight" in sat and "gate.bias" in sat:
            shared["sat_gate"] = {
                "gate.weight": sat["gate.weight"].detach().cpu(),
                "gate.bias": sat["gate.bias"].detach().cpu(),
            }

    shared_path = out_dir / "shared_frozen.pt"
    tmp = shared_path.with_suffix(".pt.tmp")
    torch.save(shared, tmp, _use_new_zipfile_serialization=False)
    tmp.replace(shared_path)

    manifest = {
        "kind": "agillm4_dblock_bench_manifest_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_ckpt": str(ckpt),
        "source_step": int(ck.get("step", 0) or 0),
        "source_seen_tok": int(ck.get("seen_tok", 0) or 0),
        "cfg": cfg,
        "tie_weights": tie_weights,
        "tokenizer_id": ck.get("tokenizer_id"),
        "vocab": vocab,
        "dblock_blocks": int(args.dblock_blocks),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "block_size": int(args.block_size),
        "shared": str(shared_path),
        "packages": [],
    }

    for idx, (worker_id, block_id) in enumerate(workers):
        layers = assignments[int(block_id)]
        if int(args.max_layers) > 0 and len(layers) > int(args.max_layers):
            _n = int(args.max_layers); _off = int(args.layer_offset) % len(layers)
            _rot = layers[_off:] + layers[:_off]
            layers = sorted(_rot[:_n])
        batch_seed = args.seed + idx * 1009
        if runtime is not None:
            ids = real_token_batches(runtime, args.source, args.steps, args.batch_size, args.block_size, batch_seed)
            data_mode = "real"
        else:
            ids = token_batches(vocab, args.steps, args.batch_size, args.block_size, batch_seed)
            data_mode = "synthetic"
        pkg = {
            "kind": "agillm4_dblock_bench_package_v1",
            "worker_id": worker_id,
            "block_id": int(block_id),
            "layers": layers,
            "cfg": cfg,
            "tie_weights": tie_weights,
            "tokenizer_id": ck.get("tokenizer_id"),
            "vocab": vocab,
            "dblock_blocks": int(args.dblock_blocks),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "block_size": int(args.block_size),
            "data_mode": data_mode,
            "source": args.source,
            "ids_batches": ids,
            "block_state": local_block_state(core, layers),
            "runtime_args": {
                "attn_backend": args.attn_backend,
                "sublinear_window": int(args.sublinear_window),
                "sublinear_stride": int(args.sublinear_stride),
                "sublinear_max_anchors": int(args.sublinear_max_anchors),
                "sublinear_chunk": int(args.sublinear_chunk),
                "sublinear_sinks": int(args.sublinear_sinks),
                "sublinear_recent_anchors": int(args.sublinear_recent_anchors),
                "sublinear_pooled_landmarks": bool(args.sublinear_pooled_landmarks),
                "dblock_objective_mode": args.objective_mode,
                "dblock_ar_prob": float(args.ar_prob),
                "dblock_sat_prob": float(args.sat_prob),
                "dblock_nat_prob": float(args.nat_prob),
                "dblock_ar_loss_tokens": int(args.ar_loss_tokens),
                "dblock_sat_loss_tokens": int(args.sat_loss_tokens),
                "dblock_nat_loss_tokens": int(args.nat_loss_tokens),
                "nat_mask_ratio": float(args.nat_mask_ratio),
                "nat_max_tokens": int(args.nat_max_tokens),
            },
        }
        out = out_dir / f"lease_{worker_id}_block{block_id}_agillm4bench.pt"
        tmp = out.with_suffix(".pt.tmp")
        torch.save(pkg, tmp, _use_new_zipfile_serialization=False)
        tmp.replace(out)
        manifest["packages"].append(
            {
                "worker_id": worker_id,
                "block_id": int(block_id),
                "layers": layers,
                "path": str(out),
                "bytes": out.stat().st_size,
            }
        )
        print(json.dumps({"event": "save_package", **manifest["packages"][-1]}), flush=True)

    manifest["wall_sec"] = round(time.time() - start, 3)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "out_dir": str(out_dir), "wall_sec": manifest["wall_sec"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
