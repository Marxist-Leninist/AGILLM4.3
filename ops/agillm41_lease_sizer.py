#!/usr/bin/env python3
"""Device-aware AGILLM4.1 lease sizer.

Given a worker's reported device + VRAM (+ RAM), return training dims so each
node trains as much real context/batch as its hardware allows, leaving ~30%
VRAM headroom for activations/optimizer. CPU nodes stay cheap; GPUs scale up.

Usage:
  agillm41_lease_sizer.py --gpu "Tesla V100-PCIE-32GB"
  agillm41_lease_sizer.py --device cpu --ram-gb 3
Prints JSON {tier, batch, block, repeat[, vram_gb]}.
"""
import argparse, json

GPU_VRAM = {  # name-substring -> GB, when VRAM not explicitly reported
    "h100": 80, "a100-80": 80, "a100": 40, "l40": 48, "a6000": 48, "a40": 48,
    "v100-pcie-32": 32, "v100-sxm2-32": 32, "v100": 16, "rtx 5090": 32, "5090": 32,
    "rtx 4090": 24, "4090": 24, "rtx 3090": 24, "3090": 24, "a10": 24,
    "rtx 4080": 16, "4080": 16, "t4": 16, "rtx 4070": 12, "3060": 12, "rtx 3080": 10,
}

def vram_for(gpu, explicit):
    if explicit and explicit > 0: return float(explicit)
    g = (gpu or "").lower()
    for k, v in GPU_VRAM.items():
        if k in g: return float(v)
    return 16.0

def size(device, gpu, vram_gb, ram_gb):
    if (device or "").lower() == "cpu" or not gpu:
        block = 256 if (ram_gb or 0) >= 4 else 128
        return {"tier": "cpu", "batch": 1, "block": block, "repeat": 256}
    v = vram_for(gpu, vram_gb)
    if   v >= 70: t = {"tier": "gpu-80g", "batch": 16, "block": 1300, "repeat": 256}
    elif v >= 40: t = {"tier": "gpu-48g", "batch": 10, "block": 1300, "repeat": 256}
    elif v >= 30: t = {"tier": "gpu-32g", "batch": 6,  "block": 1300, "repeat": 384}
    elif v >= 20: t = {"tier": "gpu-24g", "batch": 4,  "block": 1300, "repeat": 384}
    elif v >= 14: t = {"tier": "gpu-16g", "batch": 3,  "block": 1300, "repeat": 384}
    else:         t = {"tier": "gpu-sm",  "batch": 1,  "block": 1300, "repeat": 384}
    t["vram_gb"] = v
    return t

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=""); ap.add_argument("--gpu", default="")
    ap.add_argument("--vram-gb", type=float, default=0); ap.add_argument("--ram-gb", type=float, default=0)
    a = ap.parse_args()
    print(json.dumps(size(a.device, a.gpu, a.vram_gb, a.ram_gb)))
