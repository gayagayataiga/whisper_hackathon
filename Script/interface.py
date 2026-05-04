#!/usr/bin/env python3
"""
interface.py - Jetson AGX Orin 受け子サーバー

アーキテクチャ:
  Raspi
       ↓ HTTP POST(wav)
  受け子サーバー(8000) ← このファイル
       ↓ HTTP POST(wav)
  推論サーバー(8001)
       ↓ HTTP Response(text)
  受け子サーバー(8000)
       ├─ タイムスタンプ付き txt 追記保存
       └─ HTTP POST(text) → VLA サーバー

エンドポイント:
  POST /audio   : Raspi から wav を受け取り保存し、パイプラインを実行
  POST /image   : Raspi から画像を受け取り保存
  GET  /health  : サーバー死活確認
"""

import datetime
import json
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

# ============================================================
# 設定パラメータ
# ============================================================

# ── サーバー ──────────────────────────────────────────────────
RECEIVER_PORT = 8000

# ── 推論サーバー ──────────────────────────────────────────────
INFERENCE_URL     = "http://localhost:8001/transcribe"
INFERENCE_TIMEOUT = 30.0  # large-v3 の推論時間を考慮して余裕を持たせる

# ── Raspi 送信先 ──────────────────────────────────────────────
# 無線接続時: 10.27.72.53
# 有線接続時: 192.168.10.2 (詳細は docs/raspi_network.md を参照)
RASPI_URL     = "http://10.27.72.53:9000/command"
RASPI_TIMEOUT = 5.0

# ── 保存先ルート ──────────────────────────────────────────────
# Script/ の外(リポジトリ直下の data/)に保存する。
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── テキスト保存 ──────────────────────────────────────────────
TRANSCRIPT_DIR   = DATA_DIR / "transcripts"
TRANSCRIPT_FILE  = TRANSCRIPT_DIR / "transcript.txt"   # 互換: 旧フォーマット(タイムスタンプ + テキスト)
TRANSCRIPT_JSONL = TRANSCRIPT_DIR / "transcript.jsonl" # 構造化メタデータ付き

# ── 画像保存 ──────────────────────────────────────────────────
IMAGE_DIR = DATA_DIR / "images"

# ── 音声保存 ──────────────────────────────────────────────────
AUDIO_DIR = DATA_DIR / "audio"

# ============================================================
# ロガー設定
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# アプリ初期化
# ============================================================

app = FastAPI(title="Receiver Server", version="1.0.0")

# ============================================================
# 起動イベント
# ============================================================

@app.on_event("startup")
async def startup_event() -> None:
    """起動時に保存ディレクトリを作成する。"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Transcript directory: {TRANSCRIPT_DIR.resolve()}")
    logger.info(f"Image directory     : {IMAGE_DIR.resolve()}")
    logger.info(f"Audio directory     : {AUDIO_DIR.resolve()}")
    logger.info(f"Inference server: {INFERENCE_URL}")
    logger.info(f"Raspi server    : {RASPI_URL}")

# ============================================================
# ヘルパー関数
# ============================================================

def save_transcript(text: str) -> None:
    """
    文字起こし結果をタイムスタンプ付きで txt ファイルに追記する(従来フォーマット)。

    フォーマット: YYYY-MM-DD HH:MM:SS\t<テキスト>\n
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp}\t{text}\n"
    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    logger.info(f"Saved: {line.strip()}")


def save_transcript_jsonl(record: dict) -> None:
    """
    メタデータ付き文字起こし結果を JSONL に 1 行追記する。

    キー例: received_at, finished_at, duration_s, inference_s, model, language, text
    """
    with open(TRANSCRIPT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def forward_to_inference(wav_bytes: bytes, filename: str) -> dict:
    """
    wav バイト列を推論サーバーへ転送し、結果を返す。

    Args:
        wav_bytes: WAV ファイルのバイト列
        filename : 元のファイル名（ログ用）

    Returns:
        推論サーバーからのレスポンス dict
        {"text": str, "language": str, "duration_s": float}

    Raises:
        HTTPException: 推論サーバーとの通信失敗時
    """
    try:
        async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT) as client:
            response = await client.post(
                INFERENCE_URL,
                files={"file": (filename, wav_bytes, "audio/wav")},
            )
            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException:
        logger.error(f"Inference server timeout ({INFERENCE_TIMEOUT}s)")
        raise HTTPException(status_code=504, detail="Inference server timeout")

    except httpx.HTTPStatusError as e:
        logger.error(f"Inference server error: {e.response.status_code} {e.response.text}")
        raise HTTPException(
            status_code=502,
            detail=f"Inference server returned {e.response.status_code}"
        )

    except httpx.ConnectError:
        logger.error("Cannot connect to inference server")
        raise HTTPException(status_code=502, detail="Inference server unreachable")


async def send_to_raspi(text: str) -> bool:
    """
    文字起こし結果を Raspi へ HTTP POST する。

    Returns:
        送信成功なら True、失敗なら False
        （送信失敗はログに残すが、呼び出し元には例外を投げない）
    """
    try:
        async with httpx.AsyncClient(timeout=RASPI_TIMEOUT) as client:
            response = await client.post(RASPI_URL, json={"text": text})
            response.raise_for_status()
            logger.info(f"Sent to Raspi: status={response.status_code}")
            return True

    except httpx.TimeoutException:
        logger.warning(f"Raspi timeout ({RASPI_TIMEOUT}s) — skipped")
        return False

    except httpx.HTTPError as e:
        logger.warning(f"Raspi error: {e} — skipped")
        return False

# ============================================================
# エンドポイント
# ============================================================

@app.get("/health")
async def health_check() -> JSONResponse:
    """死活確認エンドポイント。"""
    return JSONResponse({"status": "ok"})


@app.post("/audio")
async def receive_audio(file: UploadFile = File(...)) -> JSONResponse:
    """
    Raspi から WAV ファイルを受け取り、パイプラインを実行する。

    処理フロー:
      1. wav 受信
      2. 推論サーバー(8001)へ転送
      3. text を受け取る
      4. txt 追記保存
      5. VLA サーバーへ送信

    Args:
        file: WAV ファイル（16kHz / 16bit / モノラル）

    Returns:
        {
            "text": "文字起こし結果",
            "duration_s": 1.23,
            "vla_sent": true
        }
    """
    # ── wav 受信 ──────────────────────────────────────────────
    received_at = datetime.datetime.now()
    wav_bytes = await file.read()
    logger.info(f"Received audio: {file.filename} ({len(wav_bytes)} bytes)")

    # ── wav 保存 ──────────────────────────────────────────────
    original = file.filename or "audio.wav"
    audio_path = AUDIO_DIR / f"{received_at.strftime('%Y%m%d_%H%M%S')}_{Path(original).name}"
    try:
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)
        logger.info(f"Saved audio: {audio_path}")
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")

    # ── 推論サーバーへ転送 ────────────────────────────────────
    result = await forward_to_inference(wav_bytes, file.filename or "audio.wav")
    finished_at = datetime.datetime.now()
    text = result.get("text", "").strip()
    duration_s = result.get("duration_s", 0.0)
    inference_s = result.get("inference_s", 0.0)
    model = result.get("model", "unknown")
    language = result.get("language", "unknown")

    logger.info(f"Transcribed: '{text}' ({duration_s:.2f}s)")

    # 空文字（無音・雑音）は保存・転送しない
    if not text:
        logger.info("Empty transcription — skipping save and VLA send")
        return JSONResponse({
            "text": "",
            "duration_s": duration_s,
            "vla_sent": False,
        })

    # ── txt / jsonl 追記保存 ──────────────────────────────────
    try:
        save_transcript(text)
        save_transcript_jsonl({
            "received_at": received_at.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(duration_s, 3),
            "inference_s": round(inference_s, 3),
            "model": model,
            "language": language,
            "text": text,
        })
    except Exception as e:
        # 保存失敗はログに残すが処理は続行する
        logger.error(f"Failed to save transcript: {e}")

    # ── Raspi へ送信 ──────────────────────────────────────────
    raspi_sent = await send_to_raspi(text)

    return JSONResponse({
        "text": text,
        "duration_s": duration_s,
        "raspi_sent": raspi_sent,
    })


@app.post("/image")
async def receive_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Raspi から画像を受け取り、タイムスタンプ付きで保存する。

    保存名: YYYYMMDD_HHMMSS_<元ファイル名>
    """
    image_bytes = await file.read()
    original = file.filename or "image.jpg"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = IMAGE_DIR / f"{timestamp}_{Path(original).name}"

    with open(save_path, "wb") as f:
        f.write(image_bytes)

    logger.info(f"Saved image: {save_path} ({len(image_bytes)} bytes)")

    return JSONResponse({
        "saved_as": str(save_path),
        "size_bytes": len(image_bytes),
        "content_type": file.content_type,
    })

# ============================================================
# エントリーポイント
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "interface:app",
        host="0.0.0.0",
        port=RECEIVER_PORT,
        log_level="info",
    )