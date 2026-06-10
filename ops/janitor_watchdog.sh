#!/usr/bin/env bash
# Keep the disk janitor alive. The janitor silently dying (with no watchdog) is
# what let the 64GB disk fill and crash-loop the trainer. This mirrors
# master_watchdog.sh: relaunch the janitor whenever it exits.
JANITOR="${JANITOR:-/workspace/agillm41_disk_janitor.sh}"
LOG="${DISK_JANITOR_LOG:-/workspace/agillm41_disk_janitor.log}"
while true; do
  echo "{\"event\":\"janitor_watchdog_launch\",\"at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG"
  bash "$JANITOR"
  echo "{\"event\":\"janitor_watchdog_exited_restarting\",\"at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG"
  sleep 10
done
