# AGILLM4.3 Latest Checkpoint + Inference Status

Updated: 2026-06-15T18:45Z

## Live Training Snapshot

- Node: Vast RTX 3090 24 GB
- Current run: B=22, context/window L=1536, DiffusionBlocks enabled
- Progress: 10,158,265,344 / 67,186,944,110 tokens (15.1194%)
- Throughput: 74,702 tok/s
- ETA from latest trainer line: about 8d 20h
- Latest stable full checkpoint: `pretrain_step01904985.pt`
- SHA256: `fa1ae6eca459e02712b4790cb63f90a6f4c560020aba663a2a085ee35e32d1ab`

## Hugging Face Artifacts

- Model repo: https://huggingface.co/OpenTransformer/AGILLM-4.3
- Latest checkpoint: https://huggingface.co/OpenTransformer/AGILLM-4.3/blob/main/training/agillm43_shared/checkpoints/full/pretrain_step01904985.pt
- Checkpoint SHA file: https://huggingface.co/OpenTransformer/AGILLM-4.3/blob/main/training/agillm43_shared/checkpoints/full/pretrain_step01904985.pt.sha256
- Latest inference run: https://huggingface.co/OpenTransformer/AGILLM-4.3/tree/main/training/agillm43_shared/inference/20260615T183623Z
- Latest inference status JSON: https://huggingface.co/OpenTransformer/AGILLM-4.3/blob/main/training/agillm43_shared/status/latest_inference.json

## Inference Smoke Test

Prompt:

```text
In one short paragraph, explain what AGILLM4.3 is doing during this training run.
```

Raw generated text from the latest checkpoint:

```text
In one short paragraph, explain what AGILLM4.3 is doing during this training run. 202, the of a  "s it' to be that you not.

and they are an way your was been.
The  ] I can in the first, but we's their, he said his. [  ] also have made her for my, she's more.

It would them him with
```

Note: this is a checkpoint-load and generation smoke test, not a quality claim. The current mid-training model still produces noisy text, but the latest checkpoint loads and runs inference successfully.

## Storage Policy

GitHub stores secret-free pointers, status, and small reproducibility notes. Checkpoints and inference artifacts are stored in Hugging Face so large binaries do not bloat the Git repository.
