#!/usr/bin/env bash
# start.sh - Launch receiver (8000) + inference (8001) in tmux
#
# Usage:
#   ./start.sh            # start
#   tmux attach -t whisper  # view logs (top: inference, bottom: receiver)
#   ./stop.sh             # stop

set -euo pipefail

SESSION="whisper"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../.venv/bin/python"

if ! command -v tmux &>/dev/null; then
    echo "[Error] tmux not found: sudo apt install tmux"
    exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[Error] Session '${SESSION}' is already running."
    echo "        To stop it: ./stop.sh"
    exit 1
fi


cd "${SCRIPT_DIR}"

# ── Max fan / lock clocks ─────────────────────────────────────
echo "[Info] Running jetson_clocks --fan ..."
sudo jetson_clocks --fan

# ── Record startup URL in .env.runtime (referenced by status.sh) ──
cat > "${SCRIPT_DIR}/.env.runtime" <<EOF
WHISPER_INFERENCE_URL=${WHISPER_INFERENCE_URL:-http://localhost:8001}
EOF
chmod 600 "${SCRIPT_DIR}/.env.runtime"

# Top pane: inference server (8001) -- start model loading first
# --workers 1: explicit to avoid sharing _previous_text across processes (uvicorn default, but kept for clarity)
tmux new-session -d -s "${SESSION}" -x 220 -y 50 \
    "${PYTHON} -m uvicorn whisper_server:app --host 0.0.0.0 --port 8001 --workers 1 --log-level info"

# Bottom pane: receiver server (8000)
# Pass environment variables explicitly inside the command string.
# (The existing tmux server's env may be stale, so we cannot rely on inheriting from the shell parent)
tmux split-window -t "${SESSION}" -v \
    "WHISPER_INFERENCE_URL='${WHISPER_INFERENCE_URL:-http://localhost:8001}' ${PYTHON} -m uvicorn interface:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info"

# Make the top pane larger (inference produces more log output)
tmux resize-pane -t "${SESSION}:0.0" -y 35

echo "[OK] Started."
echo ""
echo "  View logs : tmux attach -t ${SESSION}"
echo "  Stop      : ./stop.sh"
echo "  Health    : ./status.sh"
