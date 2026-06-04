#!/usr/bin/env python3
"""Create a weight-only AGILLM4 inference checkpoint.

The training checkpoint can contain optimizer state, scaler state, worker
updates, and other large training-only payloads. Distributed AR inference only
needs cfg, core weights, AR head, tokenizer metadata, and a small amount of
provenance. SAT/NAT heads can be retained with flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_present(dst: dict[str, Any], src: dict[str, Any], key: str) -> None:
    if key in src:
        dst[key] = src[key]


def main() -> int:
    ap = argparse.ArgumentParser(description="Export an AGILLM4 AR inference-slim checkpoint")
    ap.add_argument("--ckpt", required=True, help="Source full or delta checkpoint")
    ap.add_argument("--out", required=True, help="Destination slim checkpoint")
    ap.add_argument("--keep-sat", action="store_true", help="Keep SAT head for future SAT inference tests")
    ap.add_argument("--keep-nat", action="store_true", help="Keep NAT head when present")
    ap.add_argument("--sha256", action="store_true", help="Write OUT.sha256 sidecar")
    ap.add_argument("--report", help="Optional JSON report path")
    args = ap.parse_args()

    import torch

    src_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    state = torch.load(src_path, map_location="cpu", weights_only=False)
    load_sec = time.time() - start

    slim: dict[str, Any] = {
        "schema": "agillm4_infer_slim_v1",
        "source_checkpoint": str(src_path),
        "source_bytes": src_path.stat().st_size if src_path.exists() else None,
        "created_unix": time.time(),
    }

    if state.get("delta"):
        weights = state["weights"]
        slim["delta_source"] = True
        slim["core"] = weights["core"]
        slim["ar"] = weights["ar"]
        if args.keep_sat and "sat" in weights:
            slim["sat"] = weights["sat"]
        if args.keep_nat and "nat" in weights:
            slim["nat"] = weights["nat"]
        for key in ("cfg", "tokenizer_id", "tokenizer_json", "transformers_version", "tokenizers_version", "tie_weights", "step", "seen_tok", "phase"):
            copy_if_present(slim, state, key)
    else:
        for key in ("cfg", "core", "ar", "tokenizer_id", "tokenizer_json", "transformers_version", "tokenizers_version", "tie_weights", "step", "seen_tok", "wall_time", "phase"):
            copy_if_present(slim, state, key)
        if args.keep_sat:
            copy_if_present(slim, state, "sat")
        if args.keep_nat:
            copy_if_present(slim, state, "nat")

    missing = [key for key in ("core", "ar") if key not in slim]
    if missing:
        raise SystemExit(f"source checkpoint is missing required inference keys: {', '.join(missing)}")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    save_start = time.time()
    torch.save(slim, tmp, _use_new_zipfile_serialization=True)
    tmp.replace(out_path)
    save_sec = time.time() - save_start

    digest = None
    if args.sha256:
        digest = sha256_file(out_path)
        out_path.with_suffix(out_path.suffix + ".sha256").write_text(f"{digest}  {out_path.name}\n", encoding="utf-8")

    report = {
        "event": "agillm4_infer_slim_saved",
        "source": str(src_path),
        "out": str(out_path),
        "source_bytes": src_path.stat().st_size if src_path.exists() else None,
        "out_bytes": out_path.stat().st_size,
        "load_sec": round(load_sec, 3),
        "save_sec": round(save_sec, 3),
        "kept_keys": sorted(slim.keys()),
        "sha256": digest,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
