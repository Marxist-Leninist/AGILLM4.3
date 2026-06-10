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
# HF checkpoint uploader: the off-box backup. Its absence made the PowerStep
# divergence unrecoverable (no clean checkpoint anywhere). Never skip it.
tmux has-session -t uploader 2>/dev/null && echo "ok   uploader (already up)" || { tmux new-session -d -s uploader "cd /workspace/agillm-4 && bash upload_agillm4_checkpoints_loop.sh" && echo "up   uploader"; }
echo "--- sessions ---"; tmux ls 2>/dev/null
# Federation: hourly side-round publisher (4 Hetzner CPU nodes train assigned
# DiffusionBlocks) + puller that lands their updates in the trainer's async-merge dir.
start side_cycle  /workspace/side_cycle_watchdog.sh
start side_puller /workspace/agillm41_vast_side_update_puller.sh
