#!/usr/bin/env bash
# stop.sh - Stop the whisper tmux session

SESSION="whisper"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-session -t "${SESSION}"
    echo "[OK] Session '${SESSION}' stopped."
else
    echo "[Info] Session '${SESSION}' is not running."
fi

# ── Return fan control to nvfancontrol ───────────────────────
echo "[Info] Restarting nvfancontrol to restore fan control ..."
sudo systemctl restart nvfancontrol
