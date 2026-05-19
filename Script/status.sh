#!/usr/bin/env bash
# status.sh - Health check for both servers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== IP Addresses ==="
printf "  Local           : "
hostname -I | awk '{for(i=1;i<=NF;i++) printf "%s%s", $i, (i<NF?", ":"\n")}'
printf "  Raspi dest      : "
if [[ -f "${SCRIPT_DIR}/.env.runtime" ]]; then
    raspi_url=$(grep '^WHISPER_RASPI_URL=' "${SCRIPT_DIR}/.env.runtime" | cut -d= -f2-)
    echo "${raspi_url:-(unknown)}"
else
    echo "(not started / no .env.runtime)"
fi

echo ""
echo "=== Process check ==="
if pgrep -f "uvicorn interface" &>/dev/null; then
    echo "  receiver (8000) : running"
else
    echo "  receiver (8000) : stopped"
fi
if pgrep -f "uvicorn whisper_server" &>/dev/null; then
    echo "  inference (8001) : running"
else
    echo "  inference (8001) : stopped"
fi

echo ""
echo "=== Health check ==="
printf "  receiver (8000) : "
curl -sf --max-time 3 http://localhost:8000/health | python3 -m json.tool 2>/dev/null \
    || echo "UNREACHABLE"

printf "  inference (8001) : "
curl -sf --max-time 3 http://localhost:8001/health | python3 -m json.tool 2>/dev/null \
    || echo "UNREACHABLE"

echo ""
echo "=== Latest transcript ==="
TRANSCRIPT_JSONL="${SCRIPT_DIR}/../data/transcripts/transcript.jsonl"
TRANSCRIPT_TXT="${SCRIPT_DIR}/../data/transcripts/transcript.txt"
if [[ -s "${TRANSCRIPT_JSONL}" ]]; then
    tail -n 1 "${TRANSCRIPT_JSONL}" | python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
print(f\"  received   : {r.get('received_at','-')}\")
print(f\"  finished   : {r.get('finished_at','-')}\")
print(f\"  duration   : {r.get('duration_s','-')}s\")
print(f\"  infer time : {r.get('inference_s','-')}s\")
print(f\"  model      : {r.get('model','-')}\")
print(f\"  language   : {r.get('language','-')}\")
print(f\"  text       : {r.get('text','-')}\")
"
elif [[ -s "${TRANSCRIPT_TXT}" ]]; then
    # Only old format available
    tail -n 1 "${TRANSCRIPT_TXT}" | sed 's/^/  /'
else
    echo "  (no transcripts yet)"
fi
