#!/usr/bin/env python3
"""
stress_test.py - Faster-Whisper 全モデル × 10 回ストレステスト

全 Whisper モデルサイズについて、同一の音声ファイルを 10 回推論し
VRAM・推論時間などのメトリクスを計測して results/ に JSON として保存する。
"""

import json
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

# Script/ を sys.path に追加（modules パッケージを解決するため）
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.vram import get_vram_usage_mb
from modules.whisper_runner import load_whisper_model, transcribe_audio

# ============================================================
# 設定
# ============================================================

AUDIO_PATH   = Path(__file__).parent.parent.parent / "music" / "BASIC5000_0001.wav"
RESULTS_DIR  = Path(__file__).parent.parent.parent / "results" / "stress"
SAMPLE_RATE  = 16000
RUNS_PER_MODEL = 10

MODEL_SIZES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "kotoba-tech/kotoba-whisper-v2.0-faster",  # 日本語特化モデル
]

DEVICE       = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE     = "ja"
BEAM_SIZE    = 5


# ============================================================
# ヘルパー
# ============================================================

def load_wav_float32(path: Path) -> np.ndarray:
    """16kHz / 16bit mono WAV を float32 配列 [-1.0, 1.0] として読み込む。"""
    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2, "Expected 16-bit PCM"
        assert wf.getnchannels() == 1, "Expected mono"
        assert wf.getframerate() == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz"
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def print_separator(char: str = "─", width: int = 62) -> None:
    print(char * width)


# ============================================================
# メイン
# ============================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    audio = load_wav_float32(AUDIO_PATH)
    audio_duration_s = len(audio) / SAMPLE_RATE

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results: dict = {}

    print_separator("=")
    print("  Whisper Stress Test")
    print(f"  Audio : {AUDIO_PATH.name}  ({audio_duration_s:.2f}s)")
    print(f"  Models: {', '.join(MODEL_SIZES)}")
    print(f"  Runs  : {RUNS_PER_MODEL} per model")
    print(f"  Device: {DEVICE}  Compute: {COMPUTE_TYPE}")
    print_separator("=")

    for model_size in MODEL_SIZES:
        print(f"\n[Stress] ===== Model: {model_size} =====")
        model = load_whisper_model(model_size, DEVICE, COMPUTE_TYPE)

        vram_after_load_mb, vram_total_mb = get_vram_usage_mb()
        print(f"[Stress] Model loaded. VRAM: {vram_after_load_mb:.0f}/{vram_total_mb:.0f} MB")

        runs: list[dict] = []

        for run_idx in range(1, RUNS_PER_MODEL + 1):
            vram_before_mb, _ = get_vram_usage_mb()

            result = transcribe_audio(
                model, audio,
                language=LANGUAGE,
                initial_prompt=None,
                beam_size=BEAM_SIZE,
            )

            vram_after_mb, _ = get_vram_usage_mb()

            run_data = {
                "run": run_idx,
                "text": result.text,
                "inference_time_s": round(result.inference_time_s, 3),
                "audio_duration_s": round(result.audio_duration_s, 3),
                "rtf": round(result.rtf, 4),
                "vram_before_mb": round(vram_before_mb, 1),
                "vram_after_mb": round(vram_after_mb, 1),
                "language": result.language,
                "language_prob": round(result.language_prob, 4),
            }
            runs.append(run_data)

            print(
                f"  [{run_idx:02d}/{RUNS_PER_MODEL}] "
                f"infer={result.inference_time_s:.2f}s  "
                f"RTF={result.rtf:.2f}  "
                f"VRAM={vram_after_mb:.0f}MB  "
                f"→ {result.text[:60]!r}"
            )

        # ── per-model 集計 ──────────────────────────────────────
        times  = [r["inference_time_s"] for r in runs]
        vrams  = [r["vram_after_mb"]    for r in runs]

        summary = {
            "model_size":    model_size,
            "device":        DEVICE,
            "compute_type":  COMPUTE_TYPE,
            "audio_file":    AUDIO_PATH.name,
            "audio_duration_s": round(audio_duration_s, 3),
            "runs_count":    RUNS_PER_MODEL,
            "avg_time_s":    round(sum(times) / len(times), 3),
            "max_time_s":    round(max(times), 3),
            "min_time_s":    round(min(times), 3),
            "avg_vram_mb":   round(sum(vrams) / len(vrams), 1),
            "max_vram_mb":   round(max(vrams), 1),
            "vram_total_mb": round(vram_total_mb, 1),
            "vram_after_load_mb": round(vram_after_load_mb, 1),
            "runs":          runs,
        }
        all_results[model_size] = summary

        print_separator()
        print(
            f"  {model_size:10s}  "
            f"avg_time={summary['avg_time_s']:.2f}s  "
            f"max_time={summary['max_time_s']:.2f}s  "
            f"avg_VRAM={summary['avg_vram_mb']:.0f}MB  "
            f"max_VRAM={summary['max_vram_mb']:.0f}MB"
        )
        print_separator()

        # モデルごとに個別 JSON を保存
        safe_name = model_size.replace("/", "_").replace("-", "_")
        model_file = RESULTS_DIR / f"{timestamp}_{safe_name}.json"
        model_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[Stress] Saved → {model_file}")

        # モデルをアンロードして VRAM を解放してから次のモデルへ
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── 全モデル比較サマリー JSON ────────────────────────────────
    summary_file = RESULTS_DIR / f"{timestamp}_summary.json"
    summary_file.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2)
    )

    print(f"\n[Stress] All done. Summary → {summary_file}")
    print_separator("=")
    print(f"  {'Model':<12} {'avg_time':>10} {'max_time':>10} {'avg_VRAM':>10} {'max_VRAM':>10}")
    print_separator()
    for model_size, s in all_results.items():
        print(
            f"  {model_size:<12} "
            f"{s['avg_time_s']:>9.2f}s "
            f"{s['max_time_s']:>9.2f}s "
            f"{s['avg_vram_mb']:>8.0f}MB "
            f"{s['max_vram_mb']:>8.0f}MB"
        )
    print_separator("=")


if __name__ == "__main__":
    main()
