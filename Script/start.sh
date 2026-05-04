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

# ── 必須環境変数のチェック ──────────────────────────────────
# Raspi の IP 取り違えを防ぐためデフォルト値は持たせない。
: "${WHISPER_RASPI_URL:?WHISPER_RASPI_URL is required (例: export WHISPER_RASPI_URL=http://10.27.72.53:9000/command)}"

cd "${SCRIPT_DIR}"

# ── 起動時の URL を .env.runtime に記録 (status.sh が参照) ──
cat > "${SCRIPT_DIR}/.env.runtime" <<EOF
WHISPER_RASPI_URL=${WHISPER_RASPI_URL}
WHISPER_INFERENCE_URL=${WHISPER_INFERENCE_URL:-http://localhost:8001}
EOF
chmod 600 "${SCRIPT_DIR}/.env.runtime"

# 上ペイン: 推論サーバー(8001) — モデルロードを先に開始する
# --workers 1: _previous_text のプロセス間共有を避けるため明示(uvicorn のデフォルトでもあるが意図を残す)
tmux new-session -d -s "${SESSION}" -x 220 -y 50 \
    "${PYTHON} -m uvicorn whisper_server:app --host 0.0.0.0 --port 8001 --workers 1 --log-level info"

# 下ペイン: 受け子サーバー(8000)
# 環境変数はコマンド文字列内で明示的に渡す。
# (既存 tmux server の env は古いままになるため、シェル親プロセスからの継承に頼れない)
tmux split-window -t "${SESSION}" -v \
    "WHISPER_RASPI_URL='${WHISPER_RASPI_URL}' WHISPER_INFERENCE_URL='${WHISPER_INFERENCE_URL:-http://localhost:8001}' ${PYTHON} -m uvicorn interface:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info"

# 上ペインを大きめに（推論ログが多い）
tmux resize-pane -t "${SESSION}:0.0" -y 35

echo "[OK] 起動しました。"
echo ""
echo "  ログ確認 : tmux attach -t ${SESSION}"
echo "  停止     : ./stop.sh"
echo "  死活確認 : ./status.sh"
