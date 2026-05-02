#!/usr/bin/env python3
"""
Whisper 英語 ASR ベンチマーク（faster-whisper × LibriSpeech WER）
=================================================================
追加インストール: uv pip install --python .venv/bin/python datasets

使い方:
    python Script/benchmark/whisper_benchmark.py              # 全モデル、73サンプル
    python Script/benchmark/whisper_benchmark.py --n 10       # 10サンプルで速く確認
    python Script/benchmark/whisper_benchmark.py --models tiny.en large-v3
    python Script/benchmark/whisper_benchmark.py --audio your.wav
    python Script/benchmark/whisper_benchmark.py --list

保存先:
    results/en/{timestamp}_{model_name}.json   # モデルごと（全サンプルの REF/HYP/WER）
    results/en/{timestamp}_summary.json        # 全モデル集計
"""

import argparse
import io
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
try:
    from modules.vram import get_vram_usage_mb
except ImportError:
    def get_vram_usage_mb():
        return 0.0, 0.0
from wer_utils import calc_wer

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "en"
SAMPLE_RATE = 16000


# ─────────────────────────────────────────────
# モデル定義
# ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    model_id: str
    language: str = "en"
    description: str = ""


ALL_MODELS: list[ModelConfig] = [
    ModelConfig("tiny.en",           "tiny.en",                           description="39M  最軽量・英語特化"),
    ModelConfig("base.en",           "base.en",                           description="74M  軽量・英語特化"),
    ModelConfig("small.en",          "small.en",                          description="244M バランス・英語特化"),
    ModelConfig("medium.en",         "medium.en",                         description="769M 高精度・英語特化"),
    ModelConfig("large-v2",          "large-v2",                          description="1.5B 旧世代大規模"),
    ModelConfig("large-v3",          "large-v3",                          description="1.5B 最新世代大規模"),
    ModelConfig("turbo",             "turbo",                             description="809M large-v3蒸留・高速"),
    ModelConfig("distil-large-v3",   "distil-whisper/distil-large-v3-ct2",   description="0.8B 英語特化蒸留"),
    ModelConfig("distil-large-v3.5", "distil-whisper/distil-large-v3.5-ct2", description="0.8B 英語特化蒸留・最新"),
]


# ─────────────────────────────────────────────
# デバイス検出（torch を import しない）
# ─────────────────────────────────────────────

def _detect_device() -> tuple[str, str]:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            supported = ctranslate2.get_supported_compute_types("cuda")
            for ct in ("float16", "int8_float16", "int8"):
                if ct not in supported:
                    continue
                try:
                    from faster_whisper import WhisperModel
                    m = WhisperModel("tiny", device="cuda", compute_type=ct)
                    list(m.transcribe(np.zeros(SAMPLE_RATE * 3, np.float32), language="en")[0])
                    del m
                    return "cuda", ct
                except Exception:
                    continue
    except Exception:
        pass
    return "cpu", "float32"


DEVICE, COMPUTE_TYPE = _detect_device()


# ─────────────────────────────────────────────
# 音声デコード（av 使用・soundfile 不要）
# ─────────────────────────────────────────────

def decode_audio(src: bytes) -> np.ndarray:
    """FLAC/WAV bytes → float32 mono 16kHz"""
    import av
    chunks = []
    src_rate = None
    with av.open(io.BytesIO(src)) as container:
        stream = container.streams.audio[0]
        src_rate = stream.sample_rate
        for frame in container.decode(stream):
            arr = frame.to_ndarray()[0].astype(np.float32)
            if frame.format.name in ("s16", "s16p"):
                arr /= 32768.0
            chunks.append(arr)
    audio = np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE, np.float32)
    if src_rate and src_rate != SAMPLE_RATE:
        gcd = math.gcd(src_rate, SAMPLE_RATE)
        new_len = int(len(audio) * (SAMPLE_RATE // gcd) / (src_rate // gcd))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)), audio,
        ).astype(np.float32)
    return audio


# ─────────────────────────────────────────────
# データ準備
# ─────────────────────────────────────────────

def load_librispeech(n: int) -> list[dict]:
    try:
        from datasets import Audio, load_dataset
    except ImportError:
        print("[Error] datasets 未インストール: uv pip install --python .venv/bin/python datasets",
              file=sys.stderr)
        sys.exit(1)

    print("Loading LibriSpeech (hf-internal-testing/librispeech_asr_dummy) ...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    total = len(ds)
    n = min(n, total)
    print(f"  {n}/{total} サンプルを使用\n")

    samples = []
    for i in range(n):
        item = ds[i]
        raw  = item["audio"]
        src  = raw.get("bytes") or (open(raw["path"], "rb").read() if raw.get("path") else None)
        if src is None:
            continue
        audio = decode_audio(src)
        samples.append({
            "audio":    audio,
            "duration": len(audio) / SAMPLE_RATE,
            "ref_text": item["text"],
        })
    return samples


def load_audio_file(path: str) -> dict:
    with open(path, "rb") as f:
        audio = decode_audio(f.read())
    return {"audio": audio, "duration": len(audio) / SAMPLE_RATE, "ref_text": None}


# ─────────────────────────────────────────────
# 推論
# ─────────────────────────────────────────────

@dataclass
class ModelResult:
    model_name: str
    model_id: str
    description: str
    n_samples: int
    avg_elapsed_s: float
    avg_rtf: float
    avg_wer: Optional[float]
    avg_vram_mb: float                    # 推論中のVRAM平均（参考値、累積影響あり）
    vram_before_load_mb: float = 0.0      # モデルロード前のVRAM
    vram_after_load_mb: float = 0.0       # モデルロード後のVRAM
    model_footprint_mb: float = 0.0       # モデル単体のフットプリント = after - before
    error: Optional[str] = None
    samples: list[dict] = field(default_factory=list)


def run_model(cfg: ModelConfig, samples: list[dict], timestamp: str) -> ModelResult:
    from faster_whisper import WhisperModel

    print(f"\n{'='*60}")
    print(f"[{cfg.name}]  {cfg.description}")
    print(f"{'='*60}")
    try:
        vram_before_load, vram_total = get_vram_usage_mb()
        print(f"  Loading {cfg.model_id}  (device={DEVICE}, {COMPUTE_TYPE}) ...")
        print(f"  VRAM before load: {vram_before_load:.0f} MB")
        t0    = time.monotonic()
        model = WhisperModel(cfg.model_id, device=DEVICE, compute_type=COMPUTE_TYPE, num_workers=1)
        vram_after_load, _ = get_vram_usage_mb()
        model_footprint = vram_after_load - vram_before_load
        print(f"  Loaded in {time.monotonic()-t0:.1f}s  VRAM after load: {vram_after_load:.0f} MB  (model footprint: {model_footprint:+.0f} MB)")
        print(f"  Transcribing {len(samples)} samples ...")

        per_samples = []
        elapsed_list, rtf_list, wer_list, vram_list = [], [], [], []

        for idx, s in enumerate(samples, 1):
            t1      = time.monotonic()
            segs, _ = model.transcribe(s["audio"], language=cfg.language, beam_size=5)
            hyp     = "".join(seg.text for seg in segs).strip()
            elapsed = time.monotonic() - t1
            rtf     = elapsed / s["duration"]
            wer_val = calc_wer(hyp, s["ref_text"]) if s["ref_text"] else None
            vram_mb, _ = get_vram_usage_mb()

            elapsed_list.append(elapsed)
            rtf_list.append(rtf)
            if wer_val is not None:
                wer_list.append(wer_val)
            vram_list.append(vram_mb)

            per_samples.append({
                "idx":        idx,
                "ref":        s["ref_text"] or "",
                "hyp":        hyp,
                "wer":        round(wer_val, 4) if wer_val is not None else None,
                "elapsed_s":  round(elapsed, 3),
                "duration_s": round(s["duration"], 3),
                "rtf":        round(rtf, 4),
                "vram_mb":    round(vram_mb, 1),
            })

            if idx % 10 == 0 or idx == len(samples):
                avg_wer_so_far = sum(wer_list) / len(wer_list) if wer_list else 0
                print(f"  [{idx:3d}/{len(samples)}]  WER={avg_wer_so_far*100:.1f}%  RTF={rtf:.2f}  VRAM={vram_mb:.0f}MB")

        del model

        result = ModelResult(
            model_name          = cfg.name,
            model_id            = cfg.model_id,
            description         = cfg.description,
            n_samples           = len(samples),
            avg_elapsed_s       = sum(elapsed_list) / len(elapsed_list),
            avg_rtf             = sum(rtf_list)     / len(rtf_list),
            avg_wer             = sum(wer_list)     / len(wer_list) if wer_list else None,
            avg_vram_mb         = sum(vram_list)    / len(vram_list) if vram_list else 0.0,
            vram_before_load_mb = vram_before_load,
            vram_after_load_mb  = vram_after_load,
            model_footprint_mb  = model_footprint,
            samples             = per_samples,
        )

        # モデルごとに即時保存（中断しても消えない）
        _save_model_json(result, timestamp, vram_total)
        return result

    except Exception as e:
        import traceback; traceback.print_exc()
        return ModelResult(cfg.name, cfg.model_id, cfg.description,
                           len(samples), 0, 0, None, 0, error=str(e))


# ─────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────

def _save_model_json(result: ModelResult, timestamp: str, vram_total: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = result.model_name.replace("/", "_").replace(".", "_")
    path = RESULTS_DIR / f"{timestamp}_{safe_name}.json"
    data = {
        "model_name":          result.model_name,
        "model_id":            result.model_id,
        "description":         result.description,
        "device":              DEVICE,
        "compute_type":        COMPUTE_TYPE,
        "dataset":             "librispeech_asr_dummy",
        "n_samples":           result.n_samples,
        "avg_wer_pct":         round(result.avg_wer * 100, 2) if result.avg_wer is not None else None,
        "avg_rtf":             round(result.avg_rtf, 4),
        "avg_elapsed_s":       round(result.avg_elapsed_s, 3),
        "vram_before_load_mb": round(result.vram_before_load_mb, 1),
        "vram_after_load_mb":  round(result.vram_after_load_mb, 1),
        "model_footprint_mb":  round(result.model_footprint_mb, 1),
        "avg_vram_mb":         round(result.avg_vram_mb, 1),
        "vram_total_mb":       round(vram_total, 1),
        "samples":             result.samples,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"  Saved → {path.relative_to(Path.cwd())}")


def _save_summary_json(results: list[ModelResult], timestamp: str, n_samples: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{timestamp}_summary.json"
    data = {
        "timestamp":    timestamp,
        "dataset":      "librispeech_asr_dummy",
        "n_samples":    n_samples,
        "device":       DEVICE,
        "compute_type": COMPUTE_TYPE,
        "models": [
            {
                "model_name":    r.model_name,
                "description":   r.description,
                "avg_wer_pct":   round(r.avg_wer * 100, 2) if r.avg_wer is not None else None,
                "avg_rtf":       round(r.avg_rtf, 4),
                "avg_elapsed_s": round(r.avg_elapsed_s, 3),
                "avg_vram_mb":   round(r.avg_vram_mb, 1),
                "error":         r.error,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\nSummary → {path.relative_to(Path.cwd())}")


# ─────────────────────────────────────────────
# 表示
# ─────────────────────────────────────────────

def print_results(results: list[ModelResult]) -> None:
    C = 88
    print("\n" + "=" * C)
    print("RESULTS  (WER昇順)   RTF < 1.0 = リアルタイム以上")
    print("=" * C)

    ok     = sorted([r for r in results if not r.error], key=lambda r: r.avg_wer or 999)
    failed = [r for r in results if r.error]

    print(f"{'#':<3} {'Model':<26} {'WER':>7}  {'RTF':>5}  {'avg_t(s)':>8}  {'VRAM':>7}")
    print("-" * C)
    for i, r in enumerate(ok, 1):
        wer_s  = f"{r.avg_wer * 100:>6.1f}%" if r.avg_wer is not None else "   N/A "
        vram_s = f"{r.avg_vram_mb:>5.0f}MB"  if r.avg_vram_mb         else "   N/A"
        print(f"{i:<3} {r.model_name:<26} {wer_s}  {r.avg_rtf:>5.2f}  {r.avg_elapsed_s:>8.2f}  {vram_s}")

    if failed:
        print("\n[FAILED]")
        for r in failed:
            print(f"  {r.model_name:<26}  {r.error}")
    print()


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper 英語 ASR ベンチマーク（WER + RTF + VRAM）")
    parser.add_argument("--audio",  help="自前音声ファイル（省略時は LibriSpeech）")
    parser.add_argument("--n",      type=int, default=73, help="LibriSpeech サンプル数（default: 全件 73）")
    parser.add_argument("--models", nargs="+", help="実行するモデル名（省略時は全モデル）")
    parser.add_argument("--list",   action="store_true", help="モデル一覧を表示して終了")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Name':<26} {'Model ID':<40} {'Description'}")
        print("-" * 80)
        for m in ALL_MODELS:
            print(f"{m.name:<26} {m.model_id:<40} {m.description}")
        return

    selected = ALL_MODELS
    if args.models:
        selected = [m for m in ALL_MODELS if m.name in args.models]
        if not selected:
            print(f"Unknown model. Available: {[m.name for m in ALL_MODELS]}")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nDevice  : {DEVICE}  ({COMPUTE_TYPE})")
    print(f"Models  : {len(selected)}")
    print(f"Run     : {timestamp}")

    if args.audio:
        if not Path(args.audio).exists():
            print(f"File not found: {args.audio}")
            sys.exit(1)
        samples = [load_audio_file(args.audio)]
        print(f"Audio   : {args.audio}  ({samples[0]['duration']:.1f}s)  ※WER計測なし")
    else:
        samples = load_librispeech(args.n)
        total_dur = sum(s["duration"] for s in samples)
        print(f"Samples : {len(samples)}  total={total_dur:.1f}s")

    results: list[ModelResult] = []
    for i, cfg in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}]", end="")
        r = run_model(cfg, samples, timestamp)
        results.append(r)
        if r.error:
            print(f"  ✗ {r.error}")
        else:
            wer_s = f"WER={r.avg_wer * 100:.1f}%  " if r.avg_wer is not None else ""
            print(f"  ✓ {wer_s}RTF={r.avg_rtf:.2f}  avg={r.avg_elapsed_s:.2f}s/sample")

    print_results(results)
    _save_summary_json(results, timestamp, len(samples))


if __name__ == "__main__":
    main()
