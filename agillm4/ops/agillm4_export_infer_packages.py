#!/usr/bin/env python3
"""Export AGILLM4/4.1 split packages for staged AR inference.

The coordinator package keeps embeddings, final norm, and the AR head. Each
stage package keeps only the transformer/DiffusionBlock layers owned by that
worker, which makes CPU-only nodes viable for real network inference tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


def parse_stage_spec(spec: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, span = item.rsplit(":", 1)
        start_s, end_s = span.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if end <= start:
            raise ValueError(f"bad stage range {item!r}: END must be greater than START")
        out.append((name.strip(), start, end))
    return out


def tensor_cpu_dict(src: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        value = src[key]
        out[key] = value.detach().cpu() if hasattr(value, "detach") else value
    return out


def layer_state(core: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in core.items():
        if not isinstance(key, str):
            continue
        for layer in range(start, end):
            if key.startswith(f"blocks.{layer}."):
                out[key] = value.detach().cpu() if hasattr(value, "detach") else value
                break
    return out


def copy_metadata(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in (
        "tokenizer_id",
        "tokenizer_json",
        "transformers_version",
        "tokenizers_version",
        "tie_weights",
        "step",
        "seen_tok",
        "wall_time",
        "phase",
    ):
        if key in src:
            dst[key] = src[key]


def atomic_save(torch: Any, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp, _use_new_zipfile_serialization=True)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export AGILLM4 split inference coordinator/stage packages")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stages", required=True, help="comma list like geth:0-7,mcp:7-14,prime:14-21,web:21-28")
    ap.add_argument("--coordinator-name", default="coordinator_agillm4infer.pt")
    ap.add_argument("--manifest-name", default="infer_manifest.json")
    args = ap.parse_args()

    import torch

    start_time = time.time()
    ckpt = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    stages = parse_stage_spec(args.stages)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if state.get("delta"):
        weights = state["weights"]
        state = {
            **state,
            "core": weights["core"],
            "ar": weights["ar"],
            "cfg": state.get("cfg") or {},
            "tie_weights": state.get("tie_weights", False),
        }

    cfg = dict(state["cfg"])
    core = state["core"]
    vocab = int(core["emb.weight"].shape[0])
    coordinator = {
        "schema": "agillm4_split_infer_coordinator_v1",
        "source_checkpoint": str(ckpt),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cfg": cfg,
        "vocab": vocab,
        "core": tensor_cpu_dict(core, ["emb.weight", "ln.weight", "ln.bias"]),
        "ar": {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in state.get("ar", {}).items()},
    }
    copy_metadata(coordinator, state)
    coord_path = out_dir / args.coordinator_name
    atomic_save(torch, coordinator, coord_path)

    manifest: dict[str, Any] = {
        "schema": "agillm4_split_infer_manifest_v1",
        "created_at": coordinator["created_at"],
        "source_checkpoint": str(ckpt),
        "source_step": int(state.get("step", 0) or 0),
        "source_seen_tok": int(state.get("seen_tok", 0) or 0),
        "cfg": cfg,
        "vocab": vocab,
        "coordinator": {"path": str(coord_path), "bytes": coord_path.stat().st_size},
        "stages": [],
    }

    for name, start, end in stages:
        if start < 0 or end > int(cfg["layers"]):
            raise ValueError(f"stage {name!r} range {start}-{end} outside 0-{cfg['layers']}")
        stage_payload = {
            "schema": "agillm4_split_infer_stage_v1",
            "source_checkpoint": str(ckpt),
            "created_at": coordinator["created_at"],
            "worker_id": name,
            "start_layer": start,
            "end_layer": end,
            "cfg": cfg,
            "vocab": vocab,
            "core": layer_state(core, start, end),
        }
        copy_metadata(stage_payload, state)
        out = out_dir / f"stage_{name}_{start}_{end}_agillm4infer.pt"
        atomic_save(torch, stage_payload, out)
        entry = {
            "worker_id": name,
            "start_layer": start,
            "end_layer": end,
            "path": str(out),
            "bytes": out.stat().st_size,
            "num_tensors": len(stage_payload["core"]),
        }
        manifest["stages"].append(entry)
        print(json.dumps({"event": "save_stage", **entry}), flush=True)

    manifest["wall_sec"] = round(time.time() - start_time, 3)
    manifest_path = out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "manifest": str(manifest_path), "wall_sec": manifest["wall_sec"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
