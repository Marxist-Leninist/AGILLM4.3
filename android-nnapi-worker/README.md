# AGILLM Android NNAPI Worker Prototype

Android/Termux prototype for an AGILLM4.3 smartphone federation node.

It combines:

- a Termux federation worker for AGILLM DBlock slice leases;
- a small Android companion app for presence, battery, and screen-state signals;
- safety gates for heat, battery, recent activity, and manual sleep/awake state;
- NNAPI probes and micro-benchmarks for Qualcomm `qti-dsp` / `qti-default`;
- a hybrid NPU-forward / CPU-update micro training experiment.

## Device Profile

Tested on HONOR `VNE-N41`, Qualcomm SM4350 / Snapdragon 480 Plus.

Observed NNAPI devices:

- `qti-default`
- `qti-dsp`
- `qti-gpu`
- `nnapi-reference`

Best path so far: batched quantized dense graphs through `qti-default` or `qti-dsp`.

## Main Local Components

The local source commit on the Android device is:

```text
87cf0cc Publish AGILLM Android NNAPI worker prototype
```

Source folder on device:

```text
/data/data/com.termux/files/home/agillm-fed-android
```

Key files in the local commit:

- `android_fed_worker.py` - user-aware Android federation worker.
- `agillm_accel_profile.py` - runs NNAPI probes and writes `logs/accel_profile.json`.
- `micro_dblock_task.py` - CPU PyTorch micro diffusion-block denoise task.
- `nnapi_probe.c` - NNAPI device enumeration.
- `nnapi_op_probe_dyn.c` - op support/compile probe.
- `nnapi_dsp_bench.c` - quant8 elementwise benchmark.
- `nnapi_fc_bench.c` - quant8 fully-connected benchmark.
- `nnapi_mlp_bench.c` - chained MLP-style dense graph benchmark.
- `nnapi_train_fc_micro.c` - NPU-forward / CPU-update training path.
- `nnapi_micro_agillm43_denoise.c` - micro denoise inference graph.
- `companion/` - Android presence companion app source.

## Safety Model

The worker defers local training when:

- battery is low;
- battery or device thermal sensors are over configured thresholds;
- Android/user activity is recent;
- manual state is `awake`;
- optional charging policy is enabled and device is unplugged.

Manual `sleep` overrides presence/activity, but not thermal or battery safety.

## Practical Training Finding

NNAPI does not expose full autograd/backprop. The workable Android NPU path is:

```text
NPU/NNAPI: batched quantized forward dense/denoise blocks
CPU: loss, update rule, optimizer, scheduling, federation I/O
```

A real NPU-forward/CPU-update micro training loop was tested successfully with dynamic weights and no per-step graph recompilation.
