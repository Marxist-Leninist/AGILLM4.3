#!/usr/bin/env bash
# agillm41_disk_janitor.sh - keep the 4090 training disk from filling.
# Conservative: removes ONLY orphaned temp writes + already-processed/stale side
# artifacts and checkpoints beyond the trainer's keep-count. Never touches live
# checkpoints (<10min old / open via fuser), the resume/seed deltas, or the
# newest 2 full checkpoints. Run under janitor_watchdog.sh so it can't silently die.
set -uo pipefail
WS=/workspace
SIDE="$WS/agillm41_side_updates"
ROUNDS="$WS/agillm41_side_rounds"
LOG="$WS/agillm41_disk_janitor.log"
FREE_FLOOR_GB=8
INCOMING_KEEP=4
KEEP_ROUNDS=2
UPLOAD_FLOOR_GB=22
CKPTS="$WS/agillm4_4090_ckpts"
log(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }
free_gb(){ df -k / | awk 'NR==2{print int($4/1024/1024)}'; }
prune_for_uploads(){
  newest_full=$(ls -1t "$CKPTS"/pretrain_step*.pt 2>/dev/null | head -1)
  if [ -n "$newest_full" ]; then
    for d in "$CKPTS"/pretrain_delta_step*.pt; do
      [ -e "$d" ] || continue
      [ "$d" -ot "$newest_full" ] && { rm -f "$d" && log "rm superseded-delta $d"; }
    done
  fi
  find "$WS" -maxdepth 3 -type f -name '*.bak.*' -mtime +1 -delete 2>/dev/null
  if [ "$(free_gb)" -lt "$UPLOAD_FLOOR_GB" ]; then
    ls -1t "$CKPTS"/pretrain_step*.pt 2>/dev/null | tail -n +3 | while read -r f; do
      [ -z "$(find "$f" -mmin -10 2>/dev/null)" ] && rm -f "$f" && log "rm old-ckpt(uploadfloor) $f"
    done
  fi
}
janitor_once(){
  prune_for_uploads
  find "$WS" -type f -name '*.pt.tmp' -mmin +15 2>/dev/null | while read -r f; do
    fuser "$f" >/dev/null 2>&1 || { rm -f "$f" && log "rm orphan-tmp $f"; }
  done
  find "$SIDE/accepted" "$SIDE/rejected" -type f -mmin +10 -delete 2>/dev/null
  ls -1t "$SIDE/incoming"/*.pt 2>/dev/null | tail -n +$((INCOMING_KEEP+1)) | while read -r f; do
    [ -z "$(find "$f" -mmin -2 2>/dev/null)" ] && rm -f "$f" && log "rm stale-incoming $f"
  done
  ls -1dt "$ROUNDS"/side_cycle_*/ 2>/dev/null | tail -n +$((KEEP_ROUNDS+1)) | xargs -r rm -rf 2>/dev/null
  if [ "$(free_gb)" -lt "$FREE_FLOOR_GB" ]; then
    log "LOW DISK $(free_gb)G < ${FREE_FLOOR_GB}G -> aggressive incoming prune"
    ls -1t "$SIDE/incoming"/*.pt 2>/dev/null | tail -n +2 | while read -r f; do
      [ -z "$(find "$f" -mmin -2 2>/dev/null)" ] && rm -f "$f" && log "rm(low) $f"
    done
  fi
}
if [ "${1:-}" = "--once" ]; then janitor_once; echo "free=$(free_gb)G"; exit 0; fi
log "janitor started pid=$$ floor=${FREE_FLOOR_GB}G keep_incoming=${INCOMING_KEEP}"
while true; do janitor_once; sleep 300; done
