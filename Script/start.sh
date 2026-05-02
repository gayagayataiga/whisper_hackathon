#!/usr/bin/env bash
# start.sh - 受け子(8000) + 推論(8001) を tmux で起動
#
# 使い方:
#   ./start.sh            # 起動
#   tmux attach -t whisper  # ログを確認（上: 推論, 下: 受け子）
#   ./stop.sh             # 停止

set -euo pipefail

SESSION="whisper"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../.venv/bin/python"

if ! command -v tmux &>/dev/null; then
    echo "[Error] tmux が見つかりません: sudo apt install tmux"
    exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[Error] Session '${SESSION}' は既に起動中です。"
    echo "        停止するには: ./stop.sh"
    exit 1
fi

cd "${SCRIPT_DIR}"

# 上ペイン: 推論サーバー(8001) — モデルロードを先に開始する
tmux new-session -d -s "${SESSION}" -x 220 -y 50 \
    "${PYTHON} -m uvicorn whisper_server:app --host 0.0.0.0 --port 8001 --log-level info"

# 下ペイン: 受け子サーバー(8000)
tmux split-window -t "${SESSION}" -v \
    "${PYTHON} -m uvicorn interface:app --host 0.0.0.0 --port 8000 --log-level info"

# 上ペインを大きめに（推論ログが多い）
tmux resize-pane -t "${SESSION}:0.0" -y 35

echo "[OK] 起動しました。"
echo ""
echo "  ログ確認 : tmux attach -t ${SESSION}"
echo "  停止     : ./stop.sh"
echo "  死活確認 : ./status.sh"
