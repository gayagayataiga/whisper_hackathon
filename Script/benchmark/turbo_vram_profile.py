#!/usr/bin/env python3
"""
turbo 専用 VRAM プロファイル

Jetson AGX Orin の unified memory に対応するため、
proc RSS（実プロセスメモリ）と CUDA cudaMemGetInfo の両方を計測する。

使い方:
    python Script/benchmark/turbo_vram_profile.py
"""

import ctypes
import gc
import io
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "en"
SAMPLE_RATE = 16000
MODEL = "turbo"


# ─────────────────────────────────────────────
# メモリ計測
# ─────────────────────────────────────────────

_libcudart = None

def _cuda_used_mb() -> float:
    global _libcudart
    if _libcudart is None:
        _libcudart = ctypes.CDLL("libcudart.so.12")
    free  = ctypes.c_size_t()
    total = ctypes.c_size_t()
    _libcudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    return (total.value - free.value) / 1024 ** 2


def _proc_rss_mb() -> float:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def _sys_used_mb() -> float:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0]) / 1024
    return info["MemTotal"] - info["MemAvailable"]


def snapshot(label: str, samples: list[dict]) -> dict:
    gc.collect()
    s = {
        "label":    label,
        "rss_mb":   round(_proc_rss_mb(), 1),
        "cuda_mb":  round(_cuda_used_mb(), 1),
        "sys_mb":   round(_sys_used_mb(), 1),
        "elapsed":  round(time.monotonic() - T0, 2),
    }
    samples.append(s)
    print(f"  [{s['elapsed']:6.2f}s] {label:<28s}"
          f"  RSS={s['rss_mb']:>7.0f}MB"
          f"  CUDA={s['cuda_mb']:>7.0f}MB"
          f"  SYS={s['sys_mb']:>7.0f}MB")
    return s


# ─────────────────────────────────────────────
# 音声デコード
# ─────────────────────────────────────────────

def decode_audio(src: bytes) -> np.ndarray:
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
# メイン
# ─────────────────────────────────────────────

T0 = time.monotonic()
snapshots: list[dict] = []

print(f"=== Turbo VRAM Profile ===\n")

snapshot("initial", snapshots)

print("\nLoading libraries ...")
import av
from datasets import Audio, load_dataset
from faster_whisper import WhisperModel
snapshot("after imports", snapshots)

print("\nLoading LibriSpeech (73 samples) ...")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(decode=False))
audios = []
for i in range(len(ds)):
    raw = ds[i]["audio"]
    src = raw.get("bytes") or open(raw["path"], "rb").read()
    audios.append(decode_audio(src))
snapshot("after audio decoded", snapshots)

print(f"\nLoading Whisper turbo (compute_type=float16) ...")
snapshot("before model load", snapshots)

t_load = time.monotonic()
model = WhisperModel("turbo", device="cuda", compute_type="float16", num_workers=1)
load_time = time.monotonic() - t_load
print(f"  → Loaded in {load_time:.2f}s")
snapshot("after model load", snapshots)

print(f"\nRunning inference on {len(audios)} samples ...")
peak_rss = 0
peak_cuda = 0

for i, audio in enumerate(audios, 1):
    segs, _ = model.transcribe(audio, language="en", beam_size=5)
    list(segs)  # generatorを消費
    peak_rss  = max(peak_rss,  _proc_rss_mb())
    peak_cuda = max(peak_cuda, _cuda_used_mb())
    if i in (1, 10, 30, 50, 73):
        snapshot(f"sample {i:2d}", snapshots)

print(f"\nPeak during inference:  RSS={peak_rss:.0f}MB  CUDA={peak_cuda:.0f}MB")

snapshot("after all inference", snapshots)

print("\nDeleting model ...")
del model
gc.collect()
snapshot("after del + gc", snapshots)

# ─────────────────────────────────────────────
# 集計
# ─────────────────────────────────────────────

baseline = snapshots[0]              # initial
before_load = next(s for s in snapshots if s["label"] == "before model load")
after_load  = next(s for s in snapshots if s["label"] == "after model load")
after_run   = next(s for s in snapshots if s["label"] == "after all inference")
after_del   = next(s for s in snapshots if s["label"] == "after del + gc")

print("\n" + "=" * 70)
print("  TURBO VRAM 実測まとめ")
print("=" * 70)
print(f"  proc RSS（プロセス実メモリ ── Jetson では正確な指標）:")
print(f"    起動直後               : {baseline['rss_mb']:>7.0f} MB")
print(f"    モデルロード前         : {before_load['rss_mb']:>7.0f} MB")
print(f"    モデルロード後         : {after_load['rss_mb']:>7.0f} MB"
      f"   (+{after_load['rss_mb'] - before_load['rss_mb']:.0f} MB ← モデル本体)")
print(f"    推論ピーク             : {peak_rss:>7.0f} MB"
      f"   (+{peak_rss - before_load['rss_mb']:.0f} MB ← モデル+推論バッファ)")
print(f"    全推論終了後           : {after_run['rss_mb']:>7.0f} MB")
print(f"    del + gc 後            : {after_del['rss_mb']:>7.0f} MB"
      f"   (-{after_run['rss_mb'] - after_del['rss_mb']:.0f} MB 解放)")
print()
print(f"  CUDA cudaMemGetInfo（過小評価される ── 参考値）:")
print(f"    モデルロード後         : {after_load['cuda_mb']:>7.0f} MB"
      f"   (+{after_load['cuda_mb'] - before_load['cuda_mb']:.0f} MB)")
print(f"    推論ピーク             : {peak_cuda:>7.0f} MB"
      f"   (+{peak_cuda - before_load['cuda_mb']:.0f} MB)")
print()
print(f"  モデル turbo の正味 VRAM = {peak_rss - before_load['rss_mb']:.0f} MB")
print(f"  （= 推論ピーク RSS - ロード前 RSS）")
print("=" * 70)

# JSON 保存
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = {
    "timestamp":            ts,
    "model":                MODEL,
    "compute_type":         "float16",
    "n_samples":            len(audios),
    "load_time_s":          round(load_time, 2),
    "snapshots":            snapshots,
    "peak_rss_mb":          round(peak_rss, 1),
    "peak_cuda_mb":         round(peak_cuda, 1),
    "model_vram_rss_mb":    round(peak_rss - before_load["rss_mb"], 1),
    "model_vram_cuda_mb":   round(peak_cuda - before_load["cuda_mb"], 1),
}
out_path = RESULTS_DIR / f"{ts}_turbo_vram_profile.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\nSaved → {out_path.relative_to(Path.cwd())}")
