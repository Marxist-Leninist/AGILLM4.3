#!/usr/bin/env python3
"""Promote validated public-join results into a trusted side-update inbox.

This is intentionally separate from validation. Public worker uploads stay in
quarantine until the validator writes accepted/<lease_id>.json. This promoter
then re-checks the accepted metadata, verifies the result hash, loads the result
with torch weights_only=True, normalizes the wrapper kind expected by the live
trainer, and atomically writes one .pt update for the trusted puller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

DEFAULT_SPOOL = Path(os.environ.get("AGILLM41_LEASE_SPOOL", "/root/agillm41_public_join/spool"))
DEFAULT_OUT_DIR = Path(os.environ.get("AGILLM41_TRUSTED_UPDATE_DIR", "/root/agillm41_worker/updates"))
DEFAULT_TARGET_KIND = os.environ.get("AGILLM41_PROMOTE_TARGET_KIND", "agillm41_dblock_slice_update")
DEFAULT_MAX_BYTES = int(os.environ.get("AGILLM41_MAX_RESULT_BYTES", str(1_300_000_000)))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_tensors(obj: Any):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_tensors(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def candidate_jsons(spool: Path, lease_ids: set[str] | None) -> list[Path]:
    accepted = spool / "accepted"
    if lease_ids:
        return [accepted / f"{lease_id}.json" for lease_id in sorted(lease_ids)]
    return sorted(accepted.glob("*.json"), key=lambda p: p.stat().st_mtime)


def validate_result_object(obj: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("result is not a dict")
    block_state = obj.get("block_state")
    if not isinstance(block_state, dict) or not block_state:
        raise ValueError("missing block_state")
    if not isinstance(obj.get("cfg"), dict):
        raise ValueError("missing cfg")
    tensors = list(iter_tensors(block_state))
    if not tensors:
        raise ValueError("block_state has no tensors")
    n_params = 0
    sq = 0.0
    for tensor in tensors:
        if not torch.isfinite(tensor).all():
            raise ValueError("non-finite tensor in block_state")
        n_params += int(tensor.numel())
        sq += float(tensor.float().pow(2).sum().item())
    return {"n_tensors": len(tensors), "n_params": n_params, "update_norm": round(sq ** 0.5, 4)}


def promote_one(meta_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    meta = load_json(meta_path)
    lease_id = str(meta.get("lease_id") or meta_path.stem)
    if meta.get("state") != "accepted":
        raise ValueError(f"{lease_id}: metadata state is {meta.get('state')!r}, not 'accepted'")
    result_path = Path(str(meta.get("result_file") or ""))
    if not result_path.exists():
        raise FileNotFoundError(f"{lease_id}: result file missing: {result_path}")
    size = result_path.stat().st_size
    if size <= 0 or size > args.max_result_bytes:
        raise ValueError(f"{lease_id}: result size {size} outside allowed range")
    expected_hash = meta.get("result_sha256") or meta.get("sha256")
    actual_hash = sha256_file(result_path)
    if expected_hash and str(expected_hash) != actual_hash:
        raise ValueError(f"{lease_id}: sha256 mismatch expected={expected_hash} actual={actual_hash}")

    out_dir = Path(args.out_dir)
    promoted_dir = Path(args.promoted_dir) if args.promoted_dir else (Path(args.spool) / "promoted")
    out_name = f"public_join_{lease_id}.pt"
    out_path = out_dir / out_name
    marker_path = promoted_dir / f"{lease_id}.json"
    if (out_path.exists() or marker_path.exists()) and not args.force:
        return {"event": "already_promoted", "lease_id": lease_id, "out": str(out_path), "marker": str(marker_path)}
    if args.dry_run:
        return {"event": "would_promote", "lease_id": lease_id, "bytes": size, "sha256": actual_hash, "out": str(out_path)}

    obj = torch.load(result_path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"{lease_id}: result object is not a dict")
    stats = validate_result_object(obj)
    source_kind = obj.get("kind")
    obj["kind"] = args.target_kind
    obj["public_join"] = {
        "lease_id": lease_id,
        "node_id": meta.get("node_id"),
        "source_kind": source_kind,
        "result_sha256": actual_hash,
        "promoted_at": time.time(),
        "metadata": meta.get("metadata", {}),
        "validation": meta.get("validation", {}),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    promoted_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
    os.replace(tmp_path, out_path)
    promoted_meta = dict(meta)
    promoted_meta.update(
        {
            "state": "promoted",
            "promoted_at": time.time(),
            "promoted_file": str(out_path),
            "promoted_kind": args.target_kind,
            "source_result_sha256": actual_hash,
            "promoted_bytes": out_path.stat().st_size,
            "promotion_validation": stats,
        }
    )
    marker_path.write_text(json.dumps(promoted_meta, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "event": "promoted",
        "lease_id": lease_id,
        "out": str(out_path),
        "bytes": out_path.stat().st_size,
        "source_kind": source_kind,
        "target_kind": args.target_kind,
        **stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spool", default=str(DEFAULT_SPOOL))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--promoted-dir", default="")
    ap.add_argument("--lease-id", action="append", default=[], help="Promote a specific accepted lease id. Repeatable.")
    ap.add_argument("--target-kind", default=DEFAULT_TARGET_KIND)
    ap.add_argument("--max-promote", type=int, default=1, help="Maximum accepted results to promote per run when --lease-id is omitted.")
    ap.add_argument("--max-result-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing output/marker for the same lease id.")
    args = ap.parse_args()

    lease_ids = set(args.lease_id) if args.lease_id else None
    paths = candidate_jsons(Path(args.spool), lease_ids)
    promoted = 0
    rc = 0
    for meta_path in paths:
        if lease_ids and not meta_path.exists():
            print(json.dumps({"event": "missing_accepted_meta", "path": str(meta_path)}), flush=True)
            rc = 1
            continue
        if promoted >= args.max_promote and not lease_ids:
            break
        try:
            rec = promote_one(meta_path, args)
            print(json.dumps(rec, sort_keys=True), flush=True)
            if rec.get("event") in {"promoted", "would_promote"}:
                promoted += 1
        except Exception as exc:
            print(json.dumps({"event": "promote_error", "path": str(meta_path), "error": str(exc)}), flush=True)
            rc = 1
    if not paths:
        print(json.dumps({"event": "no_accepted_results"}), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
