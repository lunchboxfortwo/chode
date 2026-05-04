#!/bin/bash
# Master training watchdog — fixed-tree trainers only.
# Runs at most 2 trainers concurrently: 1 postflop + 1 preflop (serial within each stream).
# 7p and 8p training DROPPED.
#
# START:  cd /home/do/chode
#         setsid nohup bash solver_training/training_watchdog.sh \
#             </dev/null >>solver_training/training_watchdog.log 2>&1 &
#         echo "Watchdog PID: $!"
#
# STATUS: cat solver_training/training_watchdog.status
# LOG:    tail -f solver_training/training_watchdog.log
# STOP:   kill -- -$(cat solver_training/training_watchdog.pid)

set -uo pipefail

CHODE=/home/do/chode
TRAIN_PRE=$CHODE/solver_training/preflop_fixed_train.py
TRAIN_POST=$CHODE/solver_training/postflop_fixed_train.py
PRE_DATA=$CHODE/data/preflop_tables
POST_DATA=$CHODE/data/postflop_tables
LOG=$CHODE/solver_training/training_watchdog.log
STATUS=$CHODE/solver_training/training_watchdog.status
PID_FILE=$CHODE/solver_training/training_watchdog.pid

RESTART_DELAY=30

# ── Utilities ─────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

already_running() {
    local pattern="$1"
    pgrep -af "$pattern" 2>/dev/null | grep -v "$$" > /dev/null
}

wait_for_clear() {
    local pattern="$1" label="$2"
    local waited=0
    while already_running "$pattern"; do
        [ "$waited" -eq 0 ] && log "$label: another trainer matching '$pattern' is running — waiting"
        sleep 30
        waited=$((waited + 30))
        [ "$waited" -ge 1800 ] && { log "$label: blocked 30 min — giving up this cycle"; return 1; }
    done
    [ "$waited" -gt 0 ] && log "$label: clear after ${waited}s"
    return 0
}

set_status() {
    local msg="$(date '+%Y-%m-%d %H:%M:%S') | $*"
    echo "$msg" > "$STATUS"
    echo "$msg" >> "$LOG"
}

iters_done() {
    local f="$1"
    [ -f "$f" ] || { echo 0; return; }
    python3 -c "import json; print(json.load(open('$f')).get('iterations_done',0))" 2>/dev/null || echo 0
}

# ── RAM check ────────────────────────────────────────────────────────────────

free_ram_gb() {
    # Read available RAM + swap from /proc/meminfo (no psutil dependency)
    # Count swap at 50% value (slower, less reliable under pressure)
    python3 -c "
import re
def read_val(path, key):
    with open(path) as f:
        for line in f:
            if line.startswith(key):
                return int(re.search(r'(\d+)', line).group(1))
    return 0
mem_avail = read_val('/proc/meminfo', 'MemAvailable')  # kB
swap_free = read_val('/proc/meminfo', 'SwapFree')       # kB
print(f'{(mem_avail + 0.5 * swap_free) / 1e6:.1f}')
"
}

need_ram_gb() {
    # Return minimum free RAM needed before launching a trainer.
    # Heavier trainers (3p postflop) need more headroom.
    local np=$1 kind=$2
    if [ "$kind" = "postflop" ] && [ "$np" -ge 3 ]; then
        echo 50  # 3p+ postflop checkpoint is ~11 GB + growth to ~60 GB
    elif [ "$kind" = "postflop" ]; then
        echo 25  # 2p postflop
    else
        echo 10  # preflop
    fi
}

# ── Job runners ───────────────────────────────────────────────────────────────

# run_postflop NP TARGET
run_postflop() {
    local np=$1 target=$2
    local logfile="$POST_DATA/train_${np}p_fixed.log"
    local progress_file="$POST_DATA/${np}p_postflop_fixed_solver.progress.json"
    local min_ram; min_ram=$(need_ram_gb "$np" postflop)

    while true; do
        local done; done=$(iters_done "$progress_file")
        [ "$done" -ge "$target" ] && { log "Postflop ${np}p: COMPLETE ($done/$target)"; return 0; }

        # RAM check before launching
        local ram; ram=$(free_ram_gb)
        if python3 -c "exit(0 if float('$ram') >= $min_ram else 1)"; then
            :  # enough RAM
        else
            log "Postflop ${np}p: only ${ram} GB free, need ${min_ram} GB — sleeping 120s"
            sleep 120
            continue
        fi

        wait_for_clear "postflop_fixed_train.py --players $np " "Postflop ${np}p" || { sleep 60; continue; }

        local remaining=$(( target - done ))
        local resume_flag=; [ "$done" -gt 0 ] && resume_flag="--resume"
        log "Postflop ${np}p: launching ($done done, $remaining to go, ${ram} GB free)"

        python3 "$TRAIN_POST" \
            --players "$np" --iters "$remaining" \
            --checkpoint-every 1000000 --ram-floor-gb 8.0 \
            $resume_flag >> "$logfile" 2>&1 || true

        done=$(iters_done "$progress_file")
        log "Postflop ${np}p: process exited | progress $done/$target"
        [ "$done" -ge "$target" ] && { log "Postflop ${np}p: COMPLETE"; return 0; }

        log "Postflop ${np}p: no progress — sleeping ${RESTART_DELAY}s"
        sleep "$RESTART_DELAY"
    done
}

# run_preflop NP BB TARGET
run_preflop() {
    local np=$1 bb=$2 target=$3
    local suffix=""; [ "$bb" -ne 100 ] && suffix="_${bb}bb"
    local logfile="$PRE_DATA/train_${np}p${suffix}_fixed.log"
    local progress_file="$PRE_DATA/${np}p${suffix}_preflop_fixed_solver.progress.json"
    local min_ram; min_ram=$(need_ram_gb "$np" preflop)

    while true; do
        local done; done=$(iters_done "$progress_file")
        [ "$done" -ge "$target" ] && { log "Preflop ${np}p ${bb}bb: COMPLETE ($done/$target)"; return 0; }

        # RAM check before launching
        local ram; ram=$(free_ram_gb)
        if python3 -c "exit(0 if float('$ram') >= $min_ram else 1)"; then
            :  # enough RAM
        else
            log "Preflop ${np}p ${bb}bb: only ${ram} GB free, need ${min_ram} GB — sleeping 120s"
            sleep 120
            continue
        fi

        wait_for_clear "preflop_fixed_train.py --players $np --stack-bb $bb " "Preflop ${np}p ${bb}bb" || { sleep 60; continue; }

        local remaining=$(( target - done ))
        local resume_flag=; [ "$done" -gt 0 ] && resume_flag="--resume"
        log "Preflop ${np}p ${bb}bb: launching ($done done, $remaining to go, ${ram} GB free)"

        python3 "$TRAIN_PRE" \
            --players "$np" --stack-bb "$bb" --iters "$remaining" \
            --checkpoint-every 500000 --ram-floor-gb 8.0 \
            $resume_flag >> "$logfile" 2>&1 || true

        done=$(iters_done "$progress_file")
        log "Preflop ${np}p ${bb}bb: process exited | progress $done/$target"
        [ "$done" -ge "$target" ] && { log "Preflop ${np}p ${bb}bb: COMPLETE"; return 0; }

        log "Preflop ${np}p ${bb}bb: no progress — sleeping ${RESTART_DELAY}s"
        sleep "$RESTART_DELAY"
    done
}

# ── Streams ───────────────────────────────────────────────────────────────────

postflop_stream() {
    log "postflop stream: starting"
    run_postflop 2 2000000
    run_postflop 3 3000000
    log "postflop stream: ALL DONE"
}

preflop_stream() {
    log "preflop stream: starting (2p-6p × 30/50/75/100/150/200bb)"
    for bb in 30 50 75 100 150 200; do
        for np in 2 3 4 5 6; do
            run_preflop "$np" "$bb" 30000000
        done
    done
    log "preflop stream: ALL DONE"
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo $$ > "$PID_FILE"

cleanup() {
    trap '' SIGTERM SIGINT
    log "=== Watchdog received signal — shutting down ==="
    set_status "STOPPING"
    kill -- -$$ 2>/dev/null || true
    rm -f "$PID_FILE"
    exit 0
}
trap cleanup SIGTERM SIGINT

log "=== watchdog started (PID $$) — fixed-tree trainers ==="
cd "$CHODE"

# Remove stale tmp files
find "$PRE_DATA" "$POST_DATA" -maxdepth 1 -name '.tmp_*' -mmin +10 -delete 2>/dev/null || true

# ── Scheduling strategy ──────────────────────────────────────────────────────
# 2p postflop is light enough to run alongside preflop.
# 3p postflop needs ~60 GB RAM — must run SOLO (no concurrent preflop trainer).
# We signal the preflop stream to pause during 3p postflop via a flag file.

PAUSE_FLAG="$CHODE/.preflop_paused"

pause_preflop()   { touch "$PAUSE_FLAG"; log "Pausing preflop stream for 3p postflop"; }
resume_preflop()  { rm -f "$PAUSE_FLAG"; log "Resuming preflop stream"; }

# Modified postflop stream: pause preflop during 3p
postflop_stream_controlled() {
    log "postflop stream: starting"
    run_postflop 2 2000000
    # Before launching 3p, pause the preflop stream
    pause_preflop
    run_postflop 3 3000000
    resume_preflop
    log "postflop stream: ALL DONE"
}

# Modified preflop stream: check pause flag
run_preflop_paused() {
    local np=$1 bb=$2 target=$3
    # Wait while paused
    while [ -f "$PAUSE_FLAG" ]; do
        sleep 30
    done
    run_preflop "$np" "$bb" "$target"
}

preflop_stream_controlled() {
    log "preflop stream: starting (2p-6p × 30/50/75/100/150/200bb)"
    for bb in 30 50 75 100 150 200; do
        for np in 2 3 4 5 6; do
            run_preflop_paused "$np" "$bb" 30000000
        done
    done
    log "preflop stream: ALL DONE"
}

postflop_stream_controlled &
preflop_stream_controlled &

set_status "running: 1 postflop + 1 preflop (concurrent, 3p postflop pauses preflop)"
log "All streams launched. Waiting..."

wait || true

set_status "ALL DONE"
log "=== All training streams complete ==="
rm -f "$PID_FILE"
