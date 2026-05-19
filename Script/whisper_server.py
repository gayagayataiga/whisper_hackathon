#!/usr/bin/env python3
"""
whisper_server.py - Jetson AGX Orin inference server (port 8001)

Architecture:
  Receiver server (8000)
       ↓ HTTP POST(wav)
  Inference server (8001) ← this file
       ↓ HTTP Response(text)
  Receiver server (8000)
       ├─ Append to txt file with timestamp
       └─ HTTP POST(text) → Raspi

Endpoints:
  POST /transcribe     : Receive WAV bytes and return transcription result
  POST /reset_context  : Clear the previous text (_previous_text) used for initial_prompt
  GET  /health         : Server health check (with VRAM / context length info)
"""

import ctypes
import io
import sys
import logging
import time
import traceback
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# ============================================================
# Configuration parameters
# ============================================================

INFERENCE_PORT   = 8001

# ── Faster-Whisper ────────────────────────────────────────────
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_DEVICE     = "cuda"      # Use Jetson AGX Orin GPU
WHISPER_COMPUTE    = "float16"   # Leverage Tensor Cores
WHISPER_LANGUAGE   = "en"        # English only
WHISPER_BEAM_SIZE  = 5
WHISPER_BEST_OF    = 5
WHISPER_TEMPERATURE = 0.0
WHISPER_NO_SPEECH_THRESHOLD         = 0.6
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4

# ── Context retention ─────────────────────────────────────────
# Pass the previous recognition text to Whisper's initial_prompt to maintain accuracy for proper nouns.
# Thread-safe as long as uvicorn runs with --workers 1 (default).
INITIAL_PROMPT_MAX_CHARS = 200

# ── Audio format ──────────────────────────────────────────────
EXPECTED_SAMPLE_RATE = 16000  # Match the Raspi side

# ============================================================
# VRAM utility
# Jetson uses unified memory so NVML is not supported → call cudaMemGetInfo directly
# ============================================================

_libcudart = None


def _get_libcudart():
    global _libcudart
    if _libcudart is None:
        _libcudart = ctypes.CDLL("libcudart.so.12")
    return _libcudart


def get_vram_usage_mb() -> tuple[float, float]:
    """
    Returns:
        (used_mb, total_mb)
    """
    try:
        lib = _get_libcudart()
        free  = ctypes.c_size_t()
        total = ctypes.c_size_t()
        lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        used_mb  = (total.value - free.value) / 1024 ** 2
        total_mb = total.value / 1024 ** 2
        return used_mb, total_mb
    except Exception:
        return 0.0, 0.0

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

app = FastAPI(title="Whisper Inference Server", version="2.0.0")

whisper_model: Optional[WhisperModel] = None

# Context retention: keep the previous recognition text on the server
_previous_text: str = ""

# ============================================================
# Startup event
# ============================================================

@app.on_event("startup")
async def startup_event() -> None:
    global whisper_model
    logger.info(f"Loading Whisper {WHISPER_MODEL_SIZE} "
                f"(device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE})")
    try:
        whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
            num_workers=1,
        )
        vram_used, vram_total = get_vram_usage_mb()
        logger.info(f"Model loaded. VRAM: {vram_used:.0f} / {vram_total:.0f} MB")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        sys.exit(1)

# ============================================================
# Helper functions
# ============================================================

def wav_bytes_to_numpy(wav_bytes: bytes) -> np.ndarray:
    buf = io.BytesIO(wav_bytes)
    audio, sample_rate = sf.read(buf, dtype="float32")
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"Unexpected sample rate: {sample_rate}Hz (expected: {EXPECTED_SAMPLE_RATE}Hz)"
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio

# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check. Returns model load status and VRAM usage."""
    vram_used, vram_total = get_vram_usage_mb()
    return JSONResponse({
        "status": "ok",
        "model": WHISPER_MODEL_SIZE,
        "model_loaded": whisper_model is not None,
        "vram_used_mb": round(vram_used),
        "vram_total_mb": round(vram_total),
        "previous_text_chars": len(_previous_text),
    })


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
    """
    Receive a WAV file and return the transcription result.

    Returns:
        {
            "text": "transcription result",
            "language": "en",
            "duration_s": 1.23,
            "inference_s": 0.45,
            "rtf": 0.37,
        }
    """
    global _previous_text

    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        wav_bytes = await file.read()
        audio = wav_bytes_to_numpy(wav_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"WAV parse error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

    duration_s = len(audio) / EXPECTED_SAMPLE_RATE

    try:
        t_start = time.monotonic()
        segments, info = whisper_model.transcribe(
            audio,
            language=WHISPER_LANGUAGE,
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=WHISPER_TEMPERATURE,
            condition_on_previous_text=True,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
            initial_prompt=_previous_text if _previous_text else None,
            vad_filter=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        elapsed = time.monotonic() - t_start
        rtf = elapsed / duration_s if duration_s > 0 else float("inf")

    except Exception as e:
        logger.error(f"Whisper inference error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    vram_used, vram_total = get_vram_usage_mb()
    logger.info(
        f"'{text}' | lang={info.language} dur={duration_s:.2f}s "
        f"infer={elapsed:.2f}s RTF={rtf:.2f} VRAM={vram_used:.0f}/{vram_total:.0f}MB"
    )

    if text:
        combined = (_previous_text + " " + text).strip()
        _previous_text = combined[-INITIAL_PROMPT_MAX_CHARS:]

    return JSONResponse({
        "text": text,
        "language": info.language,
        "model": WHISPER_MODEL_SIZE,
        "duration_s": duration_s,
        "inference_s": round(elapsed, 3),
        "rtf": round(rtf, 3),
    })


@app.post("/reset_context")
async def reset_context() -> JSONResponse:
    """Clear the previous text (_previous_text) used for initial_prompt.

    Note: If /reset_context is called during a read-modify-write in /transcribe,
    there is a race condition where the reset effect is lost (because the transcribe
    side has already read the old value into a local variable that is written back later).
    This is acceptable since sequential operation driven by Raspi is assumed.
    To make it strict, introduce asyncio.Lock.
    """
    global _previous_text
    old_len = len(_previous_text)
    _previous_text = ""
    logger.info(f"Context reset (was {old_len} chars)")
    return JSONResponse({"status": "ok"})


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "whisper_server:app",
        host="0.0.0.0",
        port=INFERENCE_PORT,
        log_level="info",
    )
