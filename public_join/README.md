# AGILLM4.1 Public Join Layer

This folder has two public network modes.

## Join Scott's Network

Untrusted helpers should run only the outbound worker. It never opens SSH, never
receives coordinator credentials, verifies every downloaded lease artifact by
SHA-256, and submits results into quarantine on the coordinator.

```bash
python public_join/agillm41_join_worker.py \
  --coordinator-url https://your-agillm41-domain.example \
  --join-code "$AGILLM41_JOIN_CODE" \
  --device cpu \
  --threads 2 \
  --loop
```

The coordinator decides whether a quarantined result is accepted. Public helper
updates must never be merged into `master.pt` without a separate validator.

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
  --join-code-file ./join_code.txt
```

Add an AGILLM4.1 side-worker lease:

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

Security model:

- lease tokens are short-lived HMAC signatures bound to the package hash;
- public helpers can only download their lease package and submit one result;
- submitted results go to `quarantine/`;
- SSH and coordinator filesystem paths are not exposed to helpers;
- HTTPS is required for public binds unless `--allow-http` is used for local tests.

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
