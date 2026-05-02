#!/usr/bin/env bash
# stop.sh - whisper tmux セッションを停止する

SESSION="whisper"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-session -t "${SESSION}"
    echo "[OK] Session '${SESSION}' を停止しました。"
else
    echo "[Info] Session '${SESSION}' は起動していません。"
fi
