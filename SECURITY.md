# Security Policy

The public AGILLM 4.3 path is designed for untrusted volunteer compute.

- Volunteers connect outbound over HTTPS.
- Volunteers do not receive SSH access, API keys, private hostnames, private checkpoint paths, or merge scripts.
- Lease artifacts are SHA-256 checked before execution.
- Result uploads land in quarantine.
- A trusted validator must accept an update before it can be merged.
- Public helpers should run with least privilege in a disposable work directory.

Report issues through GitHub issues or by contacting the trusted AGILLM operator directly.
