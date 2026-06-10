#!/usr/bin/env bash
# Bring up the full AGILLM 4.2 training stack (idempotent). Vast's onstart only
# restores SSH after a box reboot, not the stack, so run this once after any reboot
# (or add it to the instance's onstart when the instance is next recreated):
#   vastai create instance <id> ... --onstart-cmd 'bash /workspace/stack_up.sh'
# Safe to run anytime: it only starts a session if it isn't already running.
start(){ tmux has-session -t "$1" 2>/dev/null && { echo "ok   $1 (already up)"; return; }; tmux new-session -d -s "$1" "$2" && echo "up   $1"; }
start master_wd    /workspace/master_watchdog.sh
start bucket_sync  /workspace/agillm41_bucket_sync_loop.sh
start disk_janitor /workspace/janitor_watchdog.sh
echo "--- sessions ---"; tmux ls 2>/dev/null
