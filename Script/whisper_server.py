#!/usr/bin/env python3
"""
whisper_server.py - Jetson AGX Orin 推論サーバー（port 8001）

アーキテクチャ:
  受け子サーバー(8000)
       ↓ HTTP POST(wav)
  推論サーバー(8001) ← このファイル
       ↓ HTTP Response(text)
  受け子サーバー(8000)
       ├─ タイムスタンプ付き txt 追記保存
       └─ HTTP POST(text) → Raspi

エンドポイント:
  POST /transcribe     : wav バイト列を受け取り文字起こし結果を返す
  POST /reset_context  : initial_prompt 用の直前テキスト(_previous_text)をクリア
  GET  /health         : サーバー死活確認（VRAM / コンテキスト長 情報付き）
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
# 設定パラメータ
# ============================================================

INFERENCE_PORT   = 8001

# ── Faster-Whisper ────────────────────────────────────────────
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_DEVICE     = "cuda"      # Jetson AGX Orin の GPU を使用
WHISPER_COMPUTE    = "float16"   # Tensor Core 活用
WHISPER_LANGUAGE   = "en"        # 英語固定
WHISPER_BEAM_SIZE  = 5
WHISPER_BEST_OF    = 5
WHISPER_TEMPERATURE = 0.0
WHISPER_NO_SPEECH_THRESHOLD         = 0.6
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4

# ── コンテキスト保持 ──────────────────────────────────────────
# Whisper の initial_prompt に直前の認識テキストを渡して固有名詞の精度を維持する。
# uvicorn --workers 1（デフォルト）で動かす限りスレッドセーフ。
INITIAL_PROMPT_MAX_CHARS = 200

# ── 音声フォーマット ──────────────────────────────────────────
EXPECTED_SAMPLE_RATE = 16000  # Raspi 側と合わせる

# ============================================================
# VRAM ユーティリティ
# Jetson は統合メモリのため NVML が非対応 → cudaMemGetInfo を直接呼ぶ
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

app = FastAPI(title="Whisper Inference Server", version="2.0.0")

whisper_model: Optional[WhisperModel] = None

# コンテキスト保持: 直前の認識テキストをサーバー内に保持する
_previous_text: str = ""

# ============================================================
# 起動イベント
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
# ヘルパー関数
# ============================================================

def wav_bytes_to_numpy(wav_bytes: bytes) -> np.ndarray:
    buf = io.BytesIO(wav_bytes)
    audio, sample_rate = sf.read(buf, dtype="float32")
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"サンプルレートが想定外: {sample_rate}Hz（期待値: {EXPECTED_SAMPLE_RATE}Hz）"
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio

# ============================================================
# エンドポイント
# ============================================================

@app.get("/health")
async def health_check() -> JSONResponse:
    """死活確認。モデルロード状態と VRAM 使用量を返す。"""
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
    WAV ファイルを受け取り、文字起こし結果を返す。

    Returns:
        {
            "text": "文字起こし結果",
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
    """initial_prompt 用の直前テキスト(_previous_text)をクリアする。

    Note: /transcribe の read-modify-write 中に /reset_context が走ると
    reset 効果が失われる race がある(transcribe 側で旧値を read 済みの
    ローカル変数が後で write back されるため)。Raspi 駆動の逐次運用前提
    なのでほぼ起きず許容。厳密にしたい場合は asyncio.Lock を導入する。
    """
    global _previous_text
    old_len = len(_previous_text)
    _previous_text = ""
    logger.info(f"Context reset (was {old_len} chars)")
    return JSONResponse({"status": "ok"})


# ============================================================
# エントリーポイント
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "whisper_server:app",
        host="0.0.0.0",
        port=INFERENCE_PORT,
        log_level="info",
    )
