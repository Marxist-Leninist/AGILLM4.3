#!/usr/bin/env bash
set -Eeuo pipefail

GETH_HOST="${AGILLM41_GETH_HOST:-root@5.75.217.57}"
GETH_KEY="${AGILLM41_GETH_KEY:-/root/.ssh/agillm41_geth_ed25519}"
REMOTE_UPDATE_DIRS="${AGILLM41_REMOTE_UPDATE_DIRS:-/root/agillm41_opportunistic/updates /root/agillm41_worker/updates}"
INCOMING_DIR="${AGILLM41_INCOMING_DIR:-/workspace/agillm41_side_updates/incoming}"
ACCEPTED_DIR="${AGILLM41_ACCEPTED_DIR:-/workspace/agillm41_side_updates/accepted}"
REJECTED_DIR="${AGILLM41_REJECTED_DIR:-/workspace/agillm41_side_updates/rejected}"
STATE_DIR="${AGILLM41_PULL_STATE_DIR:-/workspace/agillm41_side_updates/pulled}"
POLL_SEC="${AGILLM41_PULL_POLL_SEC:-300}"
MIN_BYTES="${AGILLM41_PULL_MIN_BYTES:-100000000}"

mkdir -p "$INCOMING_DIR" "$ACCEPTED_DIR" "$REJECTED_DIR" "$STATE_DIR"

copy_once() {
  local remote_dir names name src dst marker safe_dir
  for remote_dir in $REMOTE_UPDATE_DIRS; do
    names="$(ssh -i "$GETH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$GETH_HOST" \
      "find '$remote_dir' -maxdepth 1 -type f -name '*.pt' -size +$((MIN_BYTES - 1))c -printf '%f\n' 2>/dev/null | sort" || true)"
    safe_dir="$(printf '%s' "$remote_dir" | tr -c 'A-Za-z0-9_.-' '_')"
    while IFS= read -r name; do
      [ -n "$name" ] || continue
      marker="$STATE_DIR/${safe_dir}__$name.done"
      [ ! -e "$marker" ] || continue
      src="$remote_dir/$name"
      dst="$INCOMING_DIR/$name"
      if [ -e "$dst" ] || [ -e "$ACCEPTED_DIR/$name" ] || [ -e "$REJECTED_DIR/$name" ]; then
        printf '{"event":"side_update_already_seen","name":"%s","remote_dir":"%s","at":"%s"}\n' "$name" "$remote_dir" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$marker"
        continue
      fi
      scp -i "$GETH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$GETH_HOST:$src" "$dst.tmp"
      mv "$dst.tmp" "$dst"
      printf '{"event":"pulled_side_update","name":"%s","remote_dir":"%s","at":"%s","dst":"%s"}\n' "$name" "$remote_dir" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$dst" | tee "$marker"
    done <<< "$names"
  done
}

if [ "${1:-}" = "--once" ]; then
  copy_once
  exit 0
fi

while true; do
  copy_once || true
  sleep "$POLL_SEC"
done
