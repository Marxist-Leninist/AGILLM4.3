# NNAPI Bench Summary

Device: HONOR VNE-N41, Qualcomm SM4350 / Snapdragon 480 Plus.

## Device Support

- `qti-dsp`: accelerator, NNAPI feature level 30, version `1.3-22.01:build_holi`
- `qti-default`: accelerator, NNAPI feature level 30, version `1.3-22.01:build_holi`
- `qti-gpu`: GPU, NNAPI feature level 30, version `1.3-22.01:build_holi`
- `nnapi-reference`: CPU reference

## Op Support

- `qti-dsp`: supports quantized `ADD`, `RELU`, `FULLY_CONNECTED`; rejects float32 `ADD`/`RELU`.
- `qti-gpu`: supports float32 elementwise ops; rejects tested quant8 elementwise and quant8 FC.
- `qti-default`: supports tested quant8 and float32 elementwise, plus quant8 FC.

## Elementwise Result

Elementwise quant8 `ADD -> RELU` is slower on DSP because dispatch overhead dominates.

| Graph | Target | Shape | Speed |
| --- | --- | --- | --- |
| quant8 ADD+RELU | `qti-dsp` | 4096 elems | ~4.8M int ops/s |
| quant8 ADD+RELU | `nnapi-reference` | 4096 elems | ~173M int ops/s |
| quant8 ADD+RELU | `qti-dsp` | 65536 elems | ~38M int ops/s |
| quant8 ADD+RELU | `nnapi-reference` | 65536 elems | ~458M int ops/s |

## Dense Result

Quantized `FULLY_CONNECTED` is the useful path.

| Graph | Target | Shape | Speed |
| --- | --- | --- | --- |
| quant8 FC | `qti-dsp` | batch 16, 256x256 | ~0.71 GMAC/s |
| quant8 FC | `qti-default` | batch 16, 256x256 | ~0.43 GMAC/s |
| quant8 FC | `nnapi-reference` | batch 16, 256x256 | ~0.33 GMAC/s |
| quant8 FC | `qti-dsp` | batch 16, 512x512 | ~1.83 GMAC/s |
| quant8 FC | `qti-default` | batch 16, 512x512 | ~1.73 GMAC/s |
| quant8 FC | `nnapi-reference` | batch 16, 512x512 | ~1.20 GMAC/s |
| quant8 FC | `qti-dsp` | batch 8, 1024x1024 | ~2.49 GMAC/s |
| quant8 FC | `qti-default` | batch 8, 1024x1024 | ~2.62 GMAC/s |
| quant8 FC | `nnapi-reference` | batch 8, 1024x1024 | ~1.75 GMAC/s |

## Micro-Denoiser MLP Shape

The PyTorch micro denoiser mostly uses small dense matrices:

- `64 -> 128 -> 64` MLP blocks
- `64 -> 192` attention projection
- `64 -> 64` output/skip projections

Single small FC ops are too small for DSP efficiency. A fused/chained NNAPI graph does much better.

Chained quant8 MLP graph: 4 pairs of `64 -> 128 -> 64` fully-connected layers, 8 FC ops total.

| Target | Batch | Shape | Speed |
| --- | ---: | --- | ---: |
| `qti-dsp` | 32 | dim 64, hidden 128, 4 pairs | ~0.89 GMAC/s |
| `qti-dsp` | 64 | dim 64, hidden 128, 4 pairs | ~1.82 GMAC/s |
| `qti-dsp` | 128 | dim 64, hidden 128, 4 pairs | ~4.13 GMAC/s |
| `qti-default` | 128 | dim 64, hidden 128, 4 pairs | ~3.72 GMAC/s |
| `qti-dsp` | 256 | dim 64, hidden 128, 4 pairs | ~6.00 GMAC/s |

## Implication

Use NNAPI accelerator only for sufficiently heavy quantized dense/conv blocks. Keep tiny elementwise work on CPU.
