# AGILLM 4.3 Inference TPS Hardware Note - 2026-07-02

This note records the hardware and configuration behind the July 2, 2026 AR/SAT/NAT inference throughput numbers.

Do not quote these results as generic model speed. Quote them as tokens per second for this exact hardware and decode setup, and state whether checkpoint load time is excluded.

## Benchmark Context

- Host: GETH
- Environment: KVM virtual machine
- CPU: Intel Xeon Processor (Skylake, IBRS, no TSX)
- vCPU count: 16
- Threading: 1 thread per core, 16 cores, 1 socket
- RAM: 30 GiB
- Swap: 95 GiB
- OS: Linux 6.8.0-107-generic x86_64
- Python: 3.12.3
- PyTorch: 2.10.0+cu128
- CUDA available to this benchmark: false
- Inference device: CPU
- Torch / CLI CPU threads: 16
- Model: AGILLM 4.3, 1.22B parameters
- Checkpoint: `pretrain_step00002127_from00243186_20260701T0647Z.pt`
- Prompt: `The quick brown fox jumps over the lazy dog and then`
- Max new tokens: 128
- Sampling: greedy, `top_k=0`, `top_p=1.0`, `temperature=0.7`, `ignore_eos`
- Timing basis: load-excluded generation time unless explicitly marked as wall time

## Primary Generation Throughput

Use the inference process internal generation timers as the primary throughput source. These timers start after model/checkpoint loading and measure generation work directly.

| Mode | Decode settings | Internal generation seconds | Throughput |
|---|---:|---:|---:|
| AR | `--mode ar` | 26.95 s | 4.8 tok/s |
| SAT fixed | `--mode sat --no-var` | 13.90 s | 9.2 tok/s |
| NAT | `--mode nat --nat_passes 4` | 2.90 s | 44.1 tok/s |
| NAT | `--mode nat --nat_passes 2` | 1.62 s | 79.2 tok/s |
| NAT | `--mode nat --nat_passes 1` | 0.81 s | 158.0 tok/s |

## Wall-Time Sanity Check

A separate wrapper also recorded process wall time. The checkpoint load baseline on this host was approximately 67.4 seconds, but each mode was launched as a separate CLI process, so wall-minus-load is too noisy for ranking tiny NAT runs. Treat this table only as a coarse sanity check that NAT is faster than AR/SAT, not as the canonical TPS source.

| Mode | Wall seconds | Rough wall-minus-load seconds |
|---|---:|---:|
| AR 128 | 95.5 s | 28.1 s |
| SAT fixed 128 | 84.1 s | 16.7 s |
| NAT 128 pass 4 | 72.5 s | 5.1 s |
| NAT 128 pass 2 | 68.9 s | 1.5 s |
| NAT 128 pass 1 | 69.2 s | 1.8 s |

## Reporting Guidance

When adding AGILLM 4.3 inference benchmark results to README files, model cards, release notes, or Hugging Face inference folders, include:

1. Hardware: CPU/GPU model, available RAM/VRAM, and thread count.
2. Device path: CPU vs CUDA, plus whether GPU was shared with training.
3. Decode mode: AR, SAT fixed/variable, NAT, and NAT pass count.
4. Token count and prompt length.
5. Whether the number is generation-only or includes checkpoint/model load time.
6. Checkpoint filename and training provenance when available.

Short form for this run:

`GETH CPU-only, 16 vCPU Intel Xeon Skylake KVM guest, 30 GiB RAM, PyTorch 2.10, 16 CPU threads, checkpoint pretrain_step00002127_from00243186_20260701T0647Z.pt, 128 generated tokens, load-excluded generation throughput.`
