#!/usr/bin/env python3
"""
turbo-only ASR script

Transcribes with Whisper turbo (large-v3 distillation) and measures WER, RTF,
and VRAM (RSS). Lets you quickly try turbo, the recommended model in the benchmark.

Usage:
    # Evaluate on all 73 LibriSpeech samples
    python Script/benchmark/turbo_run.py

    # Reduce number of samples
    python Script/benchmark/turbo_run.py --n 10

    # Transcribe a custom audio file
    python Script/benchmark/turbo_run.py --audio path/to/audio.wav

    # Read file paths from stdin (one filename per line)
    ls *.wav | python Script/benchmark/turbo_run.py --stdin
"""

import argparse
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
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wer_utils import calc_wer

RESULTS_DIR  = Path(__file__).parent.parent.parent / "results" / "en"
SAMPLE_RATE  = 16000
MODEL_ID     = "turbo"
COMPUTE_TYPE = "float16"
DEVICE       = "cuda"


# ─────────────────────────────────────────────
# Memory measurement (supports Jetson unified memory)
# ─────────────────────────────────────────────

def proc_rss_mb() -> float:
    """Actual process memory usage (most accurate metric on Jetson)"""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


_libcudart = None
def cuda_used_mb() -> float:
    """CUDA cudaMemGetInfo (underestimates on Jetson; reference value only)"""
    global _libcudart
    try:
        if _libcudart is None:
            _libcudart = ctypes.CDLL("libcudart.so.12")
        free  = ctypes.c_size_t()
        total = ctypes.c_size_t()
        _libcudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        return (total.value - free.value) / 1024 ** 2
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# Audio decoding (using av)
# ─────────────────────────────────────────────

def decode_audio(src: bytes) -> np.ndarray:
    import av
    chunks   = []
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


def load_audio_file(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return decode_audio(f.read())


def load_librispeech(n: int) -> list[dict]:
    from datasets import Audio, load_dataset
    print(f"Loading LibriSpeech (hf-internal-testing/librispeech_asr_dummy) ...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    n  = min(n, len(ds))
    samples = []
    for i in range(n):
        item = ds[i]
        raw  = item["audio"]
        src  = raw.get("bytes") or open(raw["path"], "rb").read()
        audio = decode_audio(src)
        samples.append({
            "name":     f"librispeech_{i:03d}",
            "audio":    audio,
            "duration": len(audio) / SAMPLE_RATE,
            "ref_text": item["text"],
        })
    return samples


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper turbo-only ASR benchmark")
    parser.add_argument("--audio",  nargs="+", help="Audio files (multiple accepted)")
    parser.add_argument("--stdin",  action="store_true", help="Read file paths from stdin")
    parser.add_argument("--n",      type=int, default=73, help="Number of LibriSpeech samples (default: 73)")
    parser.add_argument("--save",   action="store_true", help="Save results as JSON under results/en/")
    args = parser.parse_args()

    # ── Determine input ──────────────────────────────────────
    audio_paths: list[str] = []
    if args.stdin:
        audio_paths = [line.strip() for line in sys.stdin if line.strip()]
    elif args.audio:
        audio_paths = args.audio

    rss_initial = proc_rss_mb()
    print(f"\nDevice         : {DEVICE} ({COMPUTE_TYPE})")
    print(f"Model          : {MODEL_ID}")
    print(f"RSS initial    : {rss_initial:.0f} MB")

    if audio_paths:
        for p in audio_paths:
            if not Path(p).exists():
                print(f"[Error] File not found: {p}", file=sys.stderr)
                sys.exit(1)
        samples = []
        for p in audio_paths:
            audio = load_audio_file(p)
            samples.append({
                "name":     Path(p).name,
                "audio":    audio,
                "duration": len(audio) / SAMPLE_RATE,
                "ref_text": None,
            })
        print(f"Audio          : {len(samples)} files  total={sum(s['duration'] for s in samples):.1f}s")
    else:
        samples = load_librispeech(args.n)
        total_dur = sum(s["duration"] for s in samples)
        print(f"Samples        : {len(samples)} (LibriSpeech)  total={total_dur:.1f}s")

    # ── Model load (measure RSS just before loading to isolate model-only footprint) ──
    from faster_whisper import WhisperModel
    gc.collect()
    rss_before_load  = proc_rss_mb()
    cuda_before_load = cuda_used_mb()
    print(f"\nLoading turbo ...")
    print(f"  RSS before load: {rss_before_load:.0f} MB")
    t0 = time.monotonic()
    model = WhisperModel(MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE, num_workers=1)
    load_time = time.monotonic() - t0
    rss_after_load = proc_rss_mb()
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  RSS after load : {rss_after_load:.0f} MB  (+{rss_after_load - rss_before_load:.0f} MB <- model body)")

    # ── Inference ────────────────────────────────────────────
    print(f"\nTranscribing {len(samples)} samples ...\n")
    print(f"  {'#':>3}  {'name':<28} {'dur':>6} {'time':>6} {'RTF':>5}  {'WER':>6}  text")
    print("  " + "-" * 110)

    results        = []
    elapsed_list   = []
    rtf_list       = []
    wer_list       = []
    peak_rss       = rss_after_load

    for i, s in enumerate(samples, 1):
        t1      = time.monotonic()
        segs, _ = model.transcribe(s["audio"], language="en", beam_size=5)
        hyp     = "".join(seg.text for seg in segs).strip()
        elapsed = time.monotonic() - t1
        rtf     = elapsed / s["duration"] if s["duration"] > 0 else 0.0
        wer_val = calc_wer(hyp, s["ref_text"]) if s["ref_text"] else None

        elapsed_list.append(elapsed)
        rtf_list.append(rtf)
        if wer_val is not None:
            wer_list.append(wer_val)

        peak_rss = max(peak_rss, proc_rss_mb())

        wer_s    = f"{wer_val*100:>5.1f}%" if wer_val is not None else "    -"
        text_s   = (hyp[:60] + "…") if len(hyp) > 60 else hyp
        print(f"  {i:>3}  {s['name']:<28} {s['duration']:>5.1f}s {elapsed:>5.2f}s {rtf:>5.2f}  {wer_s}  {text_s}")

        results.append({
            "name":       s["name"],
            "duration_s": round(s["duration"], 3),
            "elapsed_s":  round(elapsed, 3),
            "rtf":        round(rtf, 4),
            "wer":        round(wer_val, 4) if wer_val is not None else None,
            "ref":        s["ref_text"] or "",
            "hyp":        hyp,
        })

    rss_after_run = proc_rss_mb()
    cuda_peak     = cuda_used_mb()

    # ── Aggregation ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  RESULTS  ({len(samples)} samples)")
    print("=" * 70)

    avg_elapsed = sum(elapsed_list) / len(elapsed_list)
    avg_rtf     = sum(rtf_list)     / len(rtf_list)
    print(f"  Inference time : total={sum(elapsed_list):.1f}s  avg={avg_elapsed:.2f}s/sample")
    print(f"  RTF            : avg={avg_rtf:.3f}  ({1/avg_rtf:.1f}x real-time)")

    if wer_list:
        avg_wer = sum(wer_list) / len(wer_list)
        exact_n = sum(1 for w in wer_list if w == 0.0)
        print(f"  WER            : avg={avg_wer*100:.2f}%  exact={exact_n}/{len(wer_list)} ({exact_n/len(wer_list)*100:.0f}%)")

    print()
    print(f"  VRAM (RSS)   : startup={rss_initial:.0f} -> before load={rss_before_load:.0f} -> after load={rss_after_load:.0f} -> inference peak={peak_rss:.0f} MB")
    print(f"               model body  : +{rss_after_load - rss_before_load:.0f} MB (before/after load difference)")
    print(f"               inference peak: +{peak_rss - rss_before_load:.0f} MB <- practical VRAM requirement")
    print(f"  VRAM (CUDA)  : inference peak={cuda_peak:.0f} MB (underestimates on Jetson; reference value)")
    print("=" * 70)

    # ── Save ─────────────────────────────────────────────────
    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"{ts}_turbo_run.json"
        out  = {
            "timestamp":      ts,
            "model":          MODEL_ID,
            "compute_type":   COMPUTE_TYPE,
            "n_samples":      len(samples),
            "load_time_s":    round(load_time, 2),
            "avg_elapsed_s":  round(avg_elapsed, 3),
            "avg_rtf":        round(avg_rtf, 4),
            "avg_wer_pct":    round(sum(wer_list)/len(wer_list)*100, 2) if wer_list else None,
            "rss_initial_mb":     round(rss_initial, 1),
            "rss_before_load_mb": round(rss_before_load, 1),
            "rss_after_load_mb":  round(rss_after_load, 1),
            "rss_peak_mb":        round(peak_rss, 1),
            "model_body_mb":      round(rss_after_load - rss_before_load, 1),
            "model_vram_mb":      round(peak_rss - rss_before_load, 1),
            "samples":            results,
        }
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\nSaved → {path.relative_to(Path.cwd())}")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
