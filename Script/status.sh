#!/usr/bin/env bash
# status.sh - 2サーバーの死活確認

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== IP アドレス ==="
printf "  ローカル        : "
hostname -I | awk '{for(i=1;i<=NF;i++) printf "%s%s", $i, (i<NF?", ":"\n")}'
printf "  Raspi 送信先    : "
grep -E '^RASPI_URL' "${SCRIPT_DIR}/interface.py" | sed -E 's/.*"(http[^"]+)".*/\1/'

echo ""
echo "=== プロセス確認 ==="
if pgrep -f "uvicorn interface" &>/dev/null; then
    echo "  受け子  (8000) : 起動中"
else
    echo "  受け子  (8000) : 停止中"
fi
if pgrep -f "uvicorn whisper_server" &>/dev/null; then
    echo "  推論    (8001) : 起動中"
else
    echo "  推論    (8001) : 停止中"
fi

echo ""
echo "=== ヘルスチェック ==="
printf "  受け子  (8000) : "
curl -sf --max-time 3 http://localhost:8000/health | python3 -m json.tool 2>/dev/null \
    || echo "UNREACHABLE"

printf "  推論    (8001) : "
curl -sf --max-time 3 http://localhost:8001/health | python3 -m json.tool 2>/dev/null \
    || echo "UNREACHABLE"

echo ""
echo "=== 最新の文字起こし ==="
TRANSCRIPT_JSONL="${SCRIPT_DIR}/transcripts/transcript.jsonl"
TRANSCRIPT_TXT="${SCRIPT_DIR}/transcripts/transcript.txt"
if [[ -s "${TRANSCRIPT_JSONL}" ]]; then
    tail -n 1 "${TRANSCRIPT_JSONL}" | python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
print(f\"  受信     : {r.get('received_at','-')}\")
print(f\"  完了     : {r.get('finished_at','-')}\")
print(f\"  音声長   : {r.get('duration_s','-')}s\")
print(f\"  推論時間 : {r.get('inference_s','-')}s\")
print(f\"  モデル   : {r.get('model','-')}\")
print(f\"  言語     : {r.get('language','-')}\")
print(f\"  テキスト : {r.get('text','-')}\")
"
elif [[ -s "${TRANSCRIPT_TXT}" ]]; then
    # 旧フォーマットしかない場合
    tail -n 1 "${TRANSCRIPT_TXT}" | sed 's/^/  /'
else
    echo "  (まだ文字起こしはありません)"
fi
