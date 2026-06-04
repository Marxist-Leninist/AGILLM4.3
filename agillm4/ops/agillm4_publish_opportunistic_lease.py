#!/usr/bin/env python3
"""Publish a copied AGILLM4.1 benchmark lease for an optional laptop worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def find_package(export_dir: Path, worker_id: str, source_worker: str | None) -> Path:
    patterns = [f"lease_{worker_id}_block*_agillm4bench.pt"]
    if source_worker:
        patterns.append(f"lease_{source_worker}_block*_agillm4bench.pt")
    patterns.append("lease_*_block*_agillm4bench.pt")
    for pat in patterns:
        matches = sorted(export_dir.glob(pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no AGILLM4.1 lease package found in {export_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--root", default="/root/agillm41_opportunistic")
    ap.add_argument("--worker-id", default="laptop-auto")
    ap.add_argument("--source-worker", default="", help="reuse another worker package, e.g. geth")
    ap.add_argument("--rewrite-worker-id", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    export_dir = Path(args.export_dir)
    root = Path(args.root)
    current = root / "current"
    shared_src = export_dir / "shared_frozen.pt"
    manifest_src = export_dir / "manifest.json"
    if not shared_src.exists():
        raise FileNotFoundError(shared_src)
    pkg_src = find_package(export_dir, args.worker_id, args.source_worker or None)

    shared_dst = current / "shared_frozen.pt"
    lease_dst = current / f"lease_{args.worker_id}.pt"
    atomic_copy(shared_src, shared_dst)

    block_id = None
    layers = None
    if args.rewrite_worker_id:
        import torch

        pkg = torch.load(pkg_src, map_location="cpu", weights_only=False)
        pkg["worker_id"] = args.worker_id
        pkg["opportunistic"] = True
        block_id = int(pkg.get("block_id", -1))
        layers = [int(x) for x in pkg.get("layers", [])]
        tmp = lease_dst.with_suffix(".pt.tmp")
        torch.save(pkg, tmp, _use_new_zipfile_serialization=False)
        tmp.replace(lease_dst)
    else:
        atomic_copy(pkg_src, lease_dst)

    manifest = {}
    if manifest_src.exists():
        manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
    if block_id is None:
        for item in manifest.get("packages", []):
            if Path(str(item.get("path", ""))).name == pkg_src.name:
                block_id = int(item.get("block_id", -1))
                layers = item.get("layers")
                break

    sidecar = {
        "event": "agillm41_opportunistic_lease",
        "worker_id": args.worker_id,
        "source_package": str(pkg_src),
        "source_manifest": str(manifest_src) if manifest_src.exists() else None,
        "source_checkpoint": manifest.get("source_ckpt"),
        "source_step": manifest.get("source_step"),
        "block_id": block_id,
        "layers": layers,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shared_bytes": shared_dst.stat().st_size,
        "lease_bytes": lease_dst.stat().st_size,
        "policy": "optional async side-update; never block master on this lease",
    }
    sidecar_path = current / f"lease_{args.worker_id}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps(sidecar, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
