# AGILLM4.3 Public Join Layer

This folder has two public network modes:

- join Scott's AGILLM4.3 network as an untrusted outbound-only helper;
- start your own signed-lease network for your own AGILLM4.3 run.

The public coordinator for Scott's network is published here. Private SSH
routes, backend hostnames, checkpoint paths, merge scripts, validator policy,
and secrets still stay out of the public repo.

## Join Scott's Network

Published coordinator:

- `AGILLM41_COORDINATOR_URL`: `https://join.opentransformers.online`
- Health check: `https://join.opentransformers.online/health`
- `AGILLM41_JOIN_CODE`: optional. Current public mode does not require one; use it only if Scott says the coordinator is gated.

The join code is an abuse-control gate, not the security model. The real trust boundary is outbound-only workers, short-lived leases, SHA-256 artifact checks, quarantine, and trusted validation before merge.

### What A Helper Runs

By default the worker runs `--device auto`: it detects CUDA, then DirectML, else CPU, sizes threads to your machine, and reports a hardware profile (GPU name, VRAM, cores, RAM) to the coordinator. Pass `--device cuda`, `--device directml`, or `--device cpu` to force one.

Linux/macOS:

```bash
git clone https://github.com/Marxist-Leninist/AGILLM4.3.git
cd AGILLM4.3
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip torch

export AGILLM41_COORDINATOR_URL="https://join.opentransformers.online"

python public_join/agillm41_join_worker.py \
  --coordinator-url "$AGILLM41_COORDINATOR_URL" \
  --device cpu \
  --threads 2 \
  --loop
```

If Scott publishes a join code, add:

```bash
export AGILLM41_JOIN_CODE="the-current-code"
```

and pass `--join-code "$AGILLM41_JOIN_CODE"`.

Windows PowerShell:

```powershell
git clone https://github.com/Marxist-Leninist/AGILLM4.3.git
cd AGILLM4.3
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip torch

$env:AGILLM41_COORDINATOR_URL = "https://join.opentransformers.online"

.\public_join\join_scotts_network.example.ps1 -Device cpu -Threads 2
```

If Scott publishes a join code, set:

```powershell
$env:AGILLM41_JOIN_CODE = "the-current-code"
```

GPU helpers can pass `--device cuda` on Linux/Windows when CUDA PyTorch is
installed, or use a custom worker command with `--worker-cmd`. CPU is the safe
default because it works on almost anything, just slowly.

### What The Worker Does

The outbound helper:

- opens only outbound HTTPS connections;
- never opens SSH or exposes a local port;
- never receives coordinator filesystem paths or credentials;
- requests a short-lived lease;
- downloads a lease package and frozen/shared artifact;
- verifies every artifact by SHA-256 before running it;
- runs one local block-worker job;
- uploads the result to coordinator `quarantine/`.

The coordinator decides whether a quarantined result is accepted. Public helper
updates must never be merged into `master.pt` or a live checkpoint without a
separate validator.


### Operator Promotion Into A Trusted Trainer

Validation and promotion are intentionally separate steps. The validator only
moves metadata to `accepted/` after checking an untrusted upload with
`torch.load(..., weights_only=True)`, finite tensors, size limits, and norm
limits. To let a trusted trainer ingest one accepted public result, an operator
can then run:

```bash
python public_join/agillm41_promote_accepted.py \
  --spool /root/agillm41_public_join/spool \
  --out-dir /root/agillm41_worker/updates \
  --lease-id <accepted-lease-id>
```

The promoter re-checks the result SHA-256, loads with `weights_only=True`,
normalizes the wrapper kind to `agillm41_dblock_slice_update`, preserves public
join audit metadata inside the update, and writes the `.pt` atomically. Do not
run promotion as a blind loop for untrusted volunteers; keep validation policy,
rate limits, and merge cadence under trusted operator control.

### Published Join Details

```text
Coordinator URL: https://join.opentransformers.online
Health: https://join.opentransformers.online/health
Join code: optional; only needed if the coordinator is running in gated mode
Recommended command:
  python public_join/agillm41_join_worker.py --coordinator-url https://join.opentransformers.online --device cpu --threads 2 --loop
```

Everything else stays private: SSH keys, Vast/Hetzner hostnames, checkpoint
directories, merge scripts, validator thresholds, and any private dataset or
secret-bearing config.

## Start Your Own Network

Run the coordinator on a machine that can prepare lease packages, preferably
behind a real domain and TLS certificate:

```bash
python public_join/agillm41_network_host.py serve \
  --host 0.0.0.0 \
  --port 8787 \
  --public-base-url https://your-agillm41-domain.example \
  --tls-cert /etc/letsencrypt/live/your-agillm41-domain.example/fullchain.pem \
  --tls-key /etc/letsencrypt/live/your-agillm41-domain.example/privkey.pem \
```

That is the open-public mode: anyone can request a lease, but submitted results
still land in quarantine and must pass validation before they can affect a
checkpoint.

If you want a lightweight abuse gate, add a join code:

```bash
openssl rand -hex 16 > join_code.txt
python public_join/agillm41_network_host.py serve \
  --host 0.0.0.0 \
  --port 8787 \
  --public-base-url https://your-agillm41-domain.example \
  --tls-cert /etc/letsencrypt/live/your-agillm41-domain.example/fullchain.pem \
  --tls-key /etc/letsencrypt/live/your-agillm41-domain.example/privkey.pem \
  --join-code-file ./join_code.txt
```

Use a join code to reduce random internet spam, result-flooding, disk fill,
and bandwidth waste. Do not treat it as a trust boundary.

If you own a domain, point a DNS record such as `join.opentransformers.online` at the
coordinator host, open TCP 443, and put a TLS reverse proxy in front of the
Python service. Caddy is the shortest path:

```caddyfile
join.opentransformers.online {
  reverse_proxy 127.0.0.1:8787
}
```

Then run the coordinator bound to localhost:

```bash
python public_join/agillm41_network_host.py serve \
  --host 127.0.0.1 \
  --port 8787 \
  --public-base-url https://join.opentransformers.online \
  --join-code-file ./join_code.txt \
  --allow-http
```

`--allow-http` is acceptable here because Caddy terminates public HTTPS and
talks to the coordinator over local loopback only. Do not bind public HTTP to
the internet.

Add an AGILLM4.3 side-worker lease:

```bash
python public_join/agillm41_network_host.py add-lease \
  --package /path/to/lease_worker_block0_agillm4bench.pt \
  --frozen /path/to/shared_frozen.pt \
  --copy-artifacts \
  --ttl-sec 900 \
  --worker-arg device=cpu \
  --worker-arg threads=2 \
  --metadata network=agillm41-public \
  --max-result-bytes 500000000
```

Useful coordinator commands:

```bash
python public_join/agillm41_network_host.py list
find agillm41_lease_spool/quarantine -maxdepth 1 -type f -name '*.json' -print
```

Security model:

- lease tokens are short-lived HMAC signatures bound to the package hash;
- public helpers can only download their lease package and submit one result;
- submitted results go to `quarantine/`;
- SSH and coordinator filesystem paths are not exposed to helpers;
- HTTPS is required for public binds unless `--allow-http` is used for local tests.
- a join code, when enabled, is only an admission/rate-limit control;
  validation/quarantine is what protects the model.

## Local Test

Terminal 1:

```bash
printf "test-code" > join_code.txt
python public_join/agillm41_network_host.py serve \
  --host 127.0.0.1 \
  --port 8787 \
  --public-base-url http://127.0.0.1:8787 \
  --join-code-file join_code.txt
```

Terminal 2:

```bash
python public_join/agillm41_network_host.py add-lease \
  --package /path/to/lease.pt \
  --frozen /path/to/shared_frozen.pt
```

Terminal 3:

```bash
python public_join/agillm41_join_worker.py \
  --coordinator-url http://127.0.0.1:8787 \
  --join-code test-code \
  --device cpu
```


## Contribution Points

Helping train earns **contribution points**, and points will be redeemable for
**distributed inference of the latest model** (the reason to contribute is that
you get to use what you help build).

- The worker auto-generates a stable `participant_id` on first run and persists
  it in `<workdir>/participant_id.txt` (override with `--participant-id` or
  `AGILLM41_PARTICIPANT_ID`). It is an opaque token, not personal information.
- Every accepted contribution prints a `{"event": "points", ...}` line so you
  can watch your balance grow.
- Points are credited **only after server-side validation** of your submitted
  update (finite values, bounded norm, structural sanity). Invalid or junk
  uploads earn nothing. Validation loads untrusted tensors with
  `weights_only=True` only - it never executes uploaded pickles.

### Check your balance / the network

```bash
curl https://join.opentransformers.online/api/v1/points/<your-participant-id>
curl https://join.opentransformers.online/api/v1/leaderboard
curl https://join.opentransformers.online/api/v1/stats
```

Earning rate is currently `10` points per accepted contribution plus a small
per-token bonus; redemption pricing for inference is published when the inference
endpoint goes live.
