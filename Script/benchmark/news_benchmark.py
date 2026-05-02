#!/usr/bin/env python3
"""
news_benchmark.py - news_audio 全ファイルを対象にモデル別ベンチマーク

各モデルで同一の音声ファイル群を 1 回ずつ推論し、
large-v3 を正解として CER・推論速度・VRAM を集計する。

使い方:
    python Script/news_benchmark.py            # ランダム 200 ファイル
    python Script/news_benchmark.py --n 500    # ランダム N ファイル
    python Script/news_benchmark.py --all      # 全 7696 ファイル（数時間）
    python Script/news_benchmark.py --models tiny small large-v3  # モデル指定
"""

import argparse
import json
import random
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

# Script/ を sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.vram import get_vram_usage_mb
from modules.whisper_runner import load_whisper_model, transcribe_audio

# ============================================================
# 定数
# ============================================================

AUDIO_DIR    = Path(__file__).parent.parent.parent / "music" / "news_audio"
RESULTS_DIR  = Path(__file__).parent.parent.parent / "results" / "news"
SAMPLE_RATE  = 16000
SRC_RATE     = 48000    # JSUT は 48kHz

ALL_MODELS   = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "kotoba-tech/kotoba-whisper-v2.0-faster",
]
DEVICE       = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE     = "ja"
BEAM_SIZE    = 5

DEFAULT_N    = 200


# ============================================================
# 音声読み込み（48kHz → 16kHz リサンプリング）
# ============================================================

def load_wav_16k(path: Path) -> tuple[np.ndarray, float]:
    """WAV を float32 16kHz に変換して返す。duration_s も返す。"""
    with wave.open(str(path), "rb") as wf:
        src_rate = wf.getframerate()
        raw      = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if src_rate != SAMPLE_RATE:
        # polyphase リサンプリング（scipy 不要・整数比限定）
        # 48000 → 16000: 1/3 ダウンサンプル
        from_rate, to_rate = src_rate, SAMPLE_RATE
        import math
        gcd    = math.gcd(from_rate, to_rate)
        up, dn = to_rate // gcd, from_rate // gcd   # 1, 3
        # シンプルな線形補間リサンプリング（精度より速度優先）
        orig_len    = len(samples)
        new_len     = int(orig_len * up / dn)
        old_indices = np.linspace(0, orig_len - 1, new_len)
        samples     = np.interp(old_indices, np.arange(orig_len), samples).astype(np.float32)

    duration_s = len(samples) / SAMPLE_RATE
    return samples, duration_s


# ============================================================
# Levenshtein 編集距離
# ============================================================

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


# ============================================================
# メイン
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",    action="store_true", help="全ファイルを使う")
    parser.add_argument("--n",      type=int, default=DEFAULT_N, help="サンプル数（--all 非指定時）")
    parser.add_argument("--seed",   type=int, default=42,        help="乱数シード")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="使用モデルリスト（HF モデル ID も指定可）")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── ファイルリスト作成 ──────────────────────────────────────
    all_files = sorted(AUDIO_DIR.glob("*.wav"))
    if not all_files:
        sys.exit(f"[Error] No WAV files in {AUDIO_DIR}")

    if args.all:
        files = all_files
    else:
        rng   = random.Random(args.seed)
        files = rng.sample(all_files, min(args.n, len(all_files)))
        files.sort()

    n_files = len(files)
    models  = args.models

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"{timestamp}_news_bench"

    print("=" * 70)
    print("  News Audio Benchmark")
    print(f"  Files : {n_files} / {len(all_files)} total")
    print(f"  Models: {', '.join(models)}")
    print(f"  Output: results/{run_label}_*.json")
    print("=" * 70)

    # model_size → list of per-file result dicts
    model_results: dict[str, list[dict]] = {}

    for model_size in models:
        print(f"\n[Bench] ===== {model_size} =====")
        model = load_whisper_model(model_size, DEVICE, COMPUTE_TYPE)

        vram_load, vram_total = get_vram_usage_mb()
        print(f"[Bench] Loaded. VRAM: {vram_load:.0f}/{vram_total:.0f} MB")

        per_file: list[dict] = []
        total_audio_s  = 0.0
        total_infer_s  = 0.0

        for idx, wav_path in enumerate(files, 1):
            try:
                audio, dur_s = load_wav_16k(wav_path)
            except Exception as e:
                print(f"  [!] Load error {wav_path.name}: {e}", file=sys.stderr)
                continue

            result = transcribe_audio(
                model, audio,
                language=LANGUAGE,
                initial_prompt=None,
                beam_size=BEAM_SIZE,
            )

            vram_mb, _ = get_vram_usage_mb()
            total_audio_s += dur_s
            total_infer_s += result.inference_time_s

            per_file.append({
                "file":            wav_path.name,
                "text":            result.text,
                "audio_duration_s": round(dur_s, 3),
                "inference_time_s": round(result.inference_time_s, 3),
                "rtf":              round(result.rtf, 4),
                "vram_mb":          round(vram_mb, 1),
            })

            if idx % 50 == 0 or idx == n_files:
                elapsed_rtf = total_infer_s / total_audio_s if total_audio_s > 0 else 0
                print(
                    f"  [{idx:4d}/{n_files}] "
                    f"avg_infer={total_infer_s/idx:.2f}s  "
                    f"RTF={elapsed_rtf:.2f}  "
                    f"VRAM={vram_mb:.0f}MB"
                )

        model_results[model_size] = per_file

        # モデル別途中結果を保存（中断しても消えない）
        ckpt_path = RESULTS_DIR / f"{run_label}_{model_size.replace('-','_')}.json"
        ckpt_path.write_text(json.dumps({
            "model_size": model_size,
            "n_files": len(per_file),
            "files": per_file,
        }, ensure_ascii=False, indent=2))
        print(f"[Bench] Saved checkpoint → {ckpt_path.name}")

        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ============================================================
    # 集計（large-v3 が完了している場合のみ CER 算出）
    # ============================================================

    print("\n" + "=" * 70)
    print("  Aggregation")
    print("=" * 70)

    ref_model = "large-v3"
    has_ref   = ref_model in model_results

    # ファイル名 → large-v3 テキストのマップ
    ref_map: dict[str, str] = {}
    if has_ref:
        for rec in model_results[ref_model]:
            ref_map[rec["file"]] = rec["text"]
    else:
        print(f"[!] {ref_model} not in this run — CER skipped")

    summary_rows: list[dict] = []

    for model_size in models:
        recs = model_results.get(model_size, [])
        if not recs:
            continue

        times = [r["inference_time_s"] for r in recs]
        vrams = [r["vram_mb"]          for r in recs]
        durs  = [r["audio_duration_s"] for r in recs]

        avg_time  = sum(times) / len(times)
        max_time  = max(times)
        avg_rtf   = sum(r["rtf"] for r in recs) / len(recs)
        avg_vram  = sum(vrams) / len(vrams)
        max_vram  = max(vrams)

        if has_ref and model_size != ref_model:
            cers = []
            for r in recs:
                ref_text = ref_map.get(r["file"], "")
                ref_len  = max(len(ref_text), 1)
                dist     = levenshtein(r["text"], ref_text)
                cers.append(dist / ref_len)
            avg_cer  = sum(cers) / len(cers)
            char_acc = 1.0 - avg_cer
            exact_n  = sum(1 for c in cers if c == 0)
            exact_rate = exact_n / len(cers)
        elif model_size == ref_model:
            avg_cer    = 0.0
            char_acc   = 1.0
            exact_rate = 1.0
        else:
            avg_cer    = float("nan")
            char_acc   = float("nan")
            exact_rate = float("nan")

        summary_rows.append({
            "model":       model_size,
            "n_files":     len(recs),
            "avg_time_s":  round(avg_time, 3),
            "max_time_s":  round(max_time, 3),
            "avg_rtf":     round(avg_rtf, 4),
            "avg_vram_mb": round(avg_vram, 1),
            "max_vram_mb": round(max_vram, 1),
            "avg_cer":     round(avg_cer, 4) if avg_cer == avg_cer else None,
            "char_acc":    round(char_acc, 4) if char_acc == char_acc else None,
            "exact_rate":  round(exact_rate, 4) if exact_rate == exact_rate else None,
        })

    # ── 表示 ─────────────────────────────────────────────────────
    W = 92
    print(f"\n  {'Model':<38} {'正答率':>6} {'文字精度':>6} {'CER':>5}  {'avg_time':>8}  {'RTF':>5}  {'avg_VRAM':>9}")
    print("-" * W)
    for r in summary_rows:
        acc_str  = f"{r['exact_rate']*100:>5.1f}%" if r['exact_rate'] is not None else "  N/A "
        cacc_str = f"{r['char_acc']*100:>5.1f}%"   if r['char_acc']   is not None else "  N/A "
        cer_str  = f"{r['avg_cer']*100:>4.1f}%"    if r['avg_cer']    is not None else " N/A"
        print(
            f"  {r['model']:<38} {acc_str}  {cacc_str}  {cer_str}  "
            f"{r['avg_time_s']:>7.2f}s  {r['avg_rtf']:>5.2f}  {r['avg_vram_mb']:>7.0f}MB"
        )

    print("=" * W)
    print(f"\n  RTF < 1.0 = リアルタイム処理以下（{ref_model} 基準の CER）\n")

    # ── 総合スコアと推薦 ─────────────────────────────────────────
    scoreable = [r for r in summary_rows if r["char_acc"] is not None]
    if scoreable:
        print("総合スコア (文字精度 / avg_time):")
        print("-" * W)
        for r in sorted(scoreable, key=lambda x: x["char_acc"] / x["avg_time_s"], reverse=True):
            score = r["char_acc"] / r["avg_time_s"]
            print(
                f"  {r['model']:<38}  score={score:.2f}  "
                f"(文字精度={r['char_acc']*100:.1f}%  avg={r['avg_time_s']:.2f}s)"
            )
        print()

        best = max(scoreable, key=lambda x: x["char_acc"] / x["avg_time_s"])
        print(f"  ★ 推薦モデル: {best['model']}  "
              f"(文字精度={best['char_acc']*100:.1f}%  avg={best['avg_time_s']:.2f}s)")

    # ── JSON 保存 ────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"{run_label}_summary.json"
    out_data = {
        "timestamp":   timestamp,
        "n_files":     n_files,
        "seed":        args.seed,
        "use_all":     args.all,
        "models":      models,
        "reference":   ref_model,
        "summary":     summary_rows,
    }
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
    print(f"\n[Bench] Summary → {out_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
