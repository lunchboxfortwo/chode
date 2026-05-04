#!/bin/bash
# Guard script — run via cron every 5 minutes.
# Restarts training_watchdog.sh if it is not running.
CHODE=/home/do/chode
PID_FILE=$CHODE/solver_training/training_watchdog.pid
LOG=$CHODE/solver_training/training_watchdog.log

# Already running?
if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        exit 0
    fi
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] guard: training_watchdog was dead — restarting" >> "$LOG"
cd "$CHODE"
setsid nohup bash solver_training/training_watchdog.sh \
    </dev/null >>"$LOG" 2>&1 &
