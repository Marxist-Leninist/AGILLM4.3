# Android NPU-Assisted Training Path

NNAPI does not expose full autograd/backprop. The workable training path on this Android device is hybrid:

1. Compile an NNAPI forward graph.
2. Feed current activations and current quantized weights as graph inputs.
3. Execute the forward pass on `qti-default` or `qti-dsp`.
4. Compute loss/update on CPU.
5. Feed updated weights into the next NPU forward without recompiling.

Prototype source in local commit:

```text
nnapi_train_fc_micro.c
```

Local commit:

```text
87cf0cc Publish AGILLM Android NNAPI worker prototype
```

## Result

Task: micro token denoise/classification with one dynamic quant8 FC layer.

| Target | Batch | Dim | Steps | Result | Speed |
| --- | ---: | ---: | ---: | --- | ---: |
| `qti-default` | 64 | 64 | 200 | 1/64 -> 64/64 correct | ~519.6 NPU forwards/s |
| `qti-dsp` | 64 | 64 | 200 | 1/64 -> 64/64 correct | ~516.0 NPU forwards/s |
| `qti-default` | 256 | 64 | 100 | 4/256 -> 256/256 correct | ~363.5 NPU forwards/s |
| `qti-dsp` | 256 | 64 | 100 | 4/256 -> 256/256 correct | ~378.2 NPU forwards/s |

This is not full NPU training, but it is a real NPU-forward/CPU-update training loop with dynamic weights and no per-step graph recompilation.

## AGILLM4.3 Implication

Diffusion block style training is a better fit than full transformer backprop on Android:

```text
NPU:
  batched quantized forward denoise candidates
  dense/MLP subgraphs
  scoring/inference-heavy pieces

CPU:
  update rule
  optimizer or forward-only search
  federation I/O
  scheduler and safety gates
```

Forward-only or mostly-forward DBlock variants are plausible here because NNAPI forward execution works and dense batched graphs reach useful throughput.
