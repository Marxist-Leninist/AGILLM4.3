# AGILLM4.3 Public Federation Access

This public repository is safe for untrusted users and volunteer nodes. It may include public coordination information, public domains, onboarding steps, score rules, and Hugging Face artifact links. It must not include live secrets, private SSH material, API tokens, paid-provider credentials, private relay internals, or unpublished operational keys.

## Public Federation Goals

- Let outside users join AGILLM4.3 training or run compatible side networks without needing private credentials.
- Let trusted or high-scoring contributors earn access to distributed inference capacity.
- Keep public helpers sandboxed so bad or low-quality updates cannot poison master training.
- Keep the public repo useful as a neutral spec: domains, protocols, scoring, accepted artifact formats, and reproducibility notes.

## What Public Nodes May See

- Public federation domain and public API entrypoints.
- Worker bootstrap instructions using generated one-time or low-trust join tokens.
- Lease package formats, update package schemas, score/points rules, and examples.
- Links to public Hugging Face model/checkpoint/inference artifacts.
- Redacted status dashboards and aggregate metrics.

## What Public Nodes Must Not See

- HF write tokens, GitHub tokens, Vast/Hetzner provider keys, SSH private keys, private relay credentials, private admin URLs, or raw internal memory dumps.
- Master checkpoint write credentials.
- Direct destructive admin commands.
- Any secret embedded in command examples, logs, screenshots, config files, or commit history.

## Public Contribution Flow

1. A user starts a worker against the public federation domain.
2. The worker receives a bounded task or lease with a narrow scope.
3. The worker submits a signed update package, inference result, benchmark, or validation proof.
4. The coordinator scores the package for usefulness, freshness, correctness, and safety.
5. Good contributors earn points. Points can unlock higher task priority or distributed inference access.
6. Bad, stale, poisoned, malformed, or low-signal packages are quarantined or rejected.

## Public vs Private Networks

Users may run their own AGILLM-compatible network and optionally bridge into the public AGILLM4.3 federation through the same package and scoring formats. Public bridging should stay permission-minimal: accept work products, not remote control over the user's machine.

## Storage Policy

- Public GitHub: docs, schemas, public config, redacted status, and links.
- Hugging Face: checkpoints, inference artifacts, dataset/lease packages, and model cards.
- Private GitHub/vault: sensitive operations notes, private manifests, encrypted secrets, and admin-only procedures.
