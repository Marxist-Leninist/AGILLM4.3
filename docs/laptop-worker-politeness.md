# AGILLM4.3 Laptop Worker Politeness Guard

Updated: 2026-06-15

The Windows laptop federation workers should be polite to the interactive user. The local laptop is not a headless training server, and the Quadro M620 is also part of the desktop experience, so CPU/GPU worker leases must yield to keyboard/mouse activity.

## Current Local Policy

- CPU join worker waits for at least 45 seconds of user idle time before claiming a lease.
- CUDA join worker waits for at least 180 seconds of user idle time before claiming a lease.
- Workers do not claim leases while the laptop is on battery unless explicitly allowed.
- CPU worker is capped to 2 threads locally.
- CPU slices were reduced to smaller work units: 2 threads, 1 step.
- CUDA remains 1 thread, 1 step, with a longer loop delay.
- Join-worker processes run below normal priority.
- Heavy slice workers are kept at low priority, and the local watchdog may cancel known AGILLM laptop slices if the user becomes active mid-round.

## Environment Knobs

- `AGILLM41_REQUIRE_IDLE_SEC_CPU`: default local CPU idle gate.
- `AGILLM41_REQUIRE_IDLE_SEC_CUDA`: default local CUDA idle gate.
- `AGILLM41_REQUIRE_IDLE_SEC_IGPU`: optional DirectML/iGPU idle gate.
- `AGILLM41_MAX_THREADS_CPU`: local CPU thread cap.
- `AGILLM41_MAX_THREADS_CUDA`: local CUDA helper thread cap.
- `AGILLM41_ALLOW_ON_BATTERY=1`: allow laptop workers to claim work on battery.
- `AGILLM41_PRESENCE_OFF=1`: disable presence gating for true headless nodes.
- `AGILLM41_STOP_SLICES_ON_USER=0`: keep in-flight local slices running even when the user returns.

## Public Safety

This policy is public-safe. It contains no join codes, tokens, SSH material, provider API keys, or private relay credentials.
