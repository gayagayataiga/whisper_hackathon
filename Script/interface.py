#!/usr/bin/env python3
"""
interface.py - Jetson AGX Orin receiver server

Architecture:
  Raspi
       ↓ HTTP POST(wav)
  Receiver server (8000) ← this file
       ↓ HTTP POST(wav)
  Inference server (8001)
       ↓ HTTP Response(text)
  Receiver server (8000)
       ├─ Append to txt file with timestamp
       └─ HTTP POST(text) → Raspi

Endpoints:
  POST /audio          : Receive wav from Raspi, save it, and run the pipeline
  POST /image          : Receive image from Raspi and save it
  POST /reset_context  : Clear the previous text for inference server initial_prompt (forwarded)
  GET  /health         : Server health check
"""

import datetime
import json
import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

# ============================================================
# Configuration parameters
# ============================================================

# ── Server ────────────────────────────────────────────────────
RECEIVER_PORT = 8000

# ── Inference server ──────────────────────────────────────────
# WHISPER_INFERENCE_URL is the base URL (no path included). Default is the local inference server.
INFERENCE_BASE_URL = os.environ.get("WHISPER_INFERENCE_URL", "http://localhost:8001")
INFERENCE_TIMEOUT  = 30.0  # Allow generous timeout to account for large-v3 inference time

# ── Raspi destination ─────────────────────────────────────────
# WHISPER_RASPI_URL is required (no default to prevent IP mix-ups).
# See docs/raspi_network.md for IP addresses per connection method.
RASPI_URL = os.environ.get("WHISPER_RASPI_URL")  # Skip sending if not set
RASPI_TIMEOUT = 5.0

# ── Save root ─────────────────────────────────────────────────
# Save outside Script/ (in data/ at the repository root).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Text storage ──────────────────────────────────────────────
TRANSCRIPT_DIR   = DATA_DIR / "transcripts"
TRANSCRIPT_FILE  = TRANSCRIPT_DIR / "transcript.txt"   # Compatible: legacy format (timestamp + text)
TRANSCRIPT_JSONL = TRANSCRIPT_DIR / "transcript.jsonl" # With structured metadata

# ── Image storage ─────────────────────────────────────────────
IMAGE_DIR = DATA_DIR / "images"

# ── Audio storage ─────────────────────────────────────────────
AUDIO_DIR = DATA_DIR / "audio"

# ============================================================
# Logger configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# App initialization
# ============================================================

app = FastAPI(title="Receiver Server", version="1.0.0")

# ============================================================
# Startup event
# ============================================================

@app.on_event("startup")
async def startup_event() -> None:
    """Create save directories on startup."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Transcript directory: {TRANSCRIPT_DIR.resolve()}")
    logger.info(f"Image directory     : {IMAGE_DIR.resolve()}")
    logger.info(f"Audio directory     : {AUDIO_DIR.resolve()}")
    logger.info(f"Inference server: {INFERENCE_BASE_URL}")
    logger.info(f"Raspi server    : {RASPI_URL}")

# ============================================================
# Helper functions
# ============================================================

def save_transcript(text: str) -> None:
    """
    Append transcription result to a txt file with timestamp (legacy format).

    Format: YYYY-MM-DD HH:MM:SS\t<text>\n
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp}\t{text}\n"
    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    logger.info(f"Saved: {line.strip()}")


def save_transcript_jsonl(record: dict) -> None:
    """
    Append a transcription result with metadata to JSONL as one line.

    Example keys: received_at, finished_at, duration_s, inference_s, model, language, text
    """
    with open(TRANSCRIPT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def forward_to_inference(wav_bytes: bytes, filename: str) -> dict:
    """
    Forward WAV bytes to the inference server and return the result.

    Args:
        wav_bytes: WAV file bytes
        filename : original filename (for logging)

    Returns:
        Response dict from the inference server
        {"text": str, "language": str, "duration_s": float}

    Raises:
        HTTPException: on communication failure with the inference server
    """
    try:
        async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT) as client:
            response = await client.post(
                f"{INFERENCE_BASE_URL}/transcribe",
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
    HTTP POST the transcription result to Raspi.

    Returns:
        True on success, False on failure
        (failure is logged but no exception is raised to the caller)
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
# Endpoints
# ============================================================

@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


@app.post("/audio")
async def receive_audio(file: UploadFile = File(...)) -> JSONResponse:
    """
    Receive a WAV file from Raspi and run the pipeline.

    Processing flow:
      1. Receive wav
      2. Forward to inference server (8001)
      3. Receive text
      4. Append to txt file
      5. Send to Raspi

    Args:
        file: WAV file (16kHz / 16bit / mono)

    Returns:
        {
            "text": "transcription result",
            "duration_s": 1.23,
            "raspi_sent": true
        }
    """
    # ── Receive wav ───────────────────────────────────────────
    received_at = datetime.datetime.now()
    wav_bytes = await file.read()
    logger.info(f"Received audio: {file.filename} ({len(wav_bytes)} bytes)")

    # ── Save wav ──────────────────────────────────────────────
    original = file.filename or "audio.wav"
    audio_path = AUDIO_DIR / f"{received_at.strftime('%Y%m%d_%H%M%S')}_{Path(original).name}"
    try:
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)
        logger.info(f"Saved audio: {audio_path}")
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")

    # ── Forward to inference server ───────────────────────────
    result = await forward_to_inference(wav_bytes, file.filename or "audio.wav")
    finished_at = datetime.datetime.now()
    text = result.get("text", "").strip()
    duration_s = result.get("duration_s", 0.0)
    inference_s = result.get("inference_s", 0.0)
    model = result.get("model", "unknown")
    language = result.get("language", "unknown")

    logger.info(f"Transcribed: '{text}' ({duration_s:.2f}s)")

    # Empty string (silence or noise) — skip save and forward
    if not text:
        logger.info("Empty transcription — skipping save and Raspi send")
        return JSONResponse({
            "text": "",
            "duration_s": duration_s,
            "raspi_sent": False,
        })

    # ── Append to txt / jsonl ─────────────────────────────────
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
        # Log save failure but continue processing
        logger.error(f"Failed to save transcript: {e}")

    # ── Send to Raspi ─────────────────────────────────────────
    raspi_sent = await send_to_raspi(text) if RASPI_URL else False

    return JSONResponse({
        "text": text,
        "duration_s": duration_s,
        "raspi_sent": raspi_sent,
    })


@app.post("/reset_context")
async def reset_context() -> JSONResponse:
    """Clear the previous text for inference server initial_prompt (forward only)."""
    try:
        async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT) as client:
            response = await client.post(f"{INFERENCE_BASE_URL}/reset_context")
            response.raise_for_status()
            return JSONResponse(response.json())

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


@app.post("/image")
async def receive_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Receive an image from Raspi and save it with a timestamp.

    Save name: YYYYMMDD_HHMMSS_<original filename>
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
# Entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "interface:app",
        host="0.0.0.0",
        port=RECEIVER_PORT,
        log_level="info",
    )