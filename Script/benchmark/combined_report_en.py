#!/usr/bin/env python3
"""
combined_report_en.py - 英語 ASR ベンチマーク 複数ラン統合レポート

results/en/ の全 summary JSON を読み込み、
セッションをまたいだ WER・速度・総合スコアを集計する。

使い方:
    python Script/benchmark/combined_report_en.py
    python Script/benchmark/combined_report_en.py --model tiny.en large-v3 turbo
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wer_utils import calc_wer

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "en"
MODEL_ORDER = [
    "tiny.en", "base.en", "small.en", "medium.en",
    "large-v2", "large-v3", "turbo",
    "distil-large-v3", "distil-large-v3.5",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="英語 ASR 複数ラン統合レポート")
    parser.add_argument("--model", nargs="+", dest="models", help="対象モデルを絞る")
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print(f"[Error] {RESULTS_DIR} が存在しません。先に whisper_benchmark.py を実行してください。")
        return

    summary_files = sorted(RESULTS_DIR.glob("*_summary.json"))
    if not summary_files:
        print(f"[Error] {RESULTS_DIR} に summary JSON がありません。")
        return

    print(f"Found {len(summary_files)} session(s):")
    for f in summary_files:
        print(f"  {f.name}")
    print()

    # ─── 全セッションのサンプルをモデル別に集約 ──────────────────

    # model_name -> list of per-sample dicts
    all_samples: dict[str, list[dict]] = {}
    session_meta: list[dict] = []

    for sf in summary_files:
        summary = json.loads(sf.read_text())
        ts = sf.stem.replace("_summary", "")
        session_meta.append({
            "timestamp":  ts,
            "n_samples":  summary.get("n_samples", 0),
            "device":     summary.get("device", ""),
            "compute_type": summary.get("compute_type", ""),
        })

        for m in summary["models"]:
            if m.get("error"):
                continue
            name      = m["model_name"]
            safe_name = name.replace("/", "_").replace(".", "_")
            model_path = RESULTS_DIR / f"{ts}_{safe_name}.json"
            if not model_path.exists():
                continue
            data = json.loads(model_path.read_text())
            if name not in all_samples:
                all_samples[name] = []
            all_samples[name].extend(data.get("samples", []))

    # ─── モデルフィルタ ──────────────────────────────────────────

    target = args.models if args.models else MODEL_ORDER
    target = [m for m in target if m in all_samples]
    if not target:
        target = list(all_samples.keys())

    # ─── 集計 ────────────────────────────────────────────────────

    rows: list[dict] = []

    for model_name in target:
        samples = all_samples[model_name]
        n = len(samples)
        if n == 0:
            continue

        # REF/HYP から WER を再計算（whisper-normalizer 適用）
        wer_list = []
        for s in samples:
            if s.get("ref") and s.get("hyp") is not None:
                wer_list.append(calc_wer(s["hyp"], s["ref"]))

        elapsed_list = [s["elapsed_s"] for s in samples if s.get("elapsed_s")]
        rtf_list     = [s["rtf"]       for s in samples if s.get("rtf")]
        vram_list    = [s["vram_mb"]   for s in samples if s.get("vram_mb")]

        avg_wer      = sum(wer_list)     / len(wer_list)     if wer_list     else None
        avg_elapsed  = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0
        avg_rtf      = sum(rtf_list)     / len(rtf_list)     if rtf_list     else 0.0
        avg_vram     = sum(vram_list)    / len(vram_list)     if vram_list    else 0.0
        exact_n      = sum(1 for w in wer_list if w == 0.0)

        # WER 分布
        wer_pct_list = [w * 100 for w in wer_list]
        wer_p50      = sorted(wer_pct_list)[len(wer_pct_list) // 2] if wer_pct_list else 0.0
        wer_p90      = sorted(wer_pct_list)[int(len(wer_pct_list) * 0.9)] if wer_pct_list else 0.0

        rows.append({
            "model_name":  model_name,
            "n_samples":   n,
            "avg_wer_pct": round(avg_wer * 100, 2) if avg_wer is not None else None,
            "wer_p50":     round(wer_p50, 2),
            "wer_p90":     round(wer_p90, 2),
            "exact_n":     exact_n,
            "exact_pct":   round(exact_n / len(wer_list) * 100, 1) if wer_list else 0.0,
            "avg_elapsed_s": round(avg_elapsed, 3),
            "avg_rtf":     round(avg_rtf, 4),
            "avg_vram_mb": round(avg_vram, 1),
        })

    if not rows:
        print("[Error] 集計できるデータがありませんでした。")
        return

    # ─── 表示 ────────────────────────────────────────────────────

    W = 96

    print("=" * W)
    print(f"  Combined Results: {len(summary_files)} session(s)  "
          f"total samples per model: {max(r['n_samples'] for r in rows)}")
    print("=" * W)
    print(f"  {'Model':<26} {'avg WER':>8}  {'p50':>6}  {'p90':>6}  "
          f"{'完全一致':>10}  {'RTF':>6}  {'VRAM':>7}")
    print("-" * W)

    for r in sorted(rows, key=lambda x: x["avg_wer_pct"] or 999):
        wer_s    = f"{r['avg_wer_pct']:>7.2f}%" if r["avg_wer_pct"] is not None else "    N/A "
        exact_s  = f"{r['exact_n']}/{r['n_samples']} ({r['exact_pct']:.0f}%)"
        vram_s   = f"{r['avg_vram_mb']:>5.0f}MB" if r["avg_vram_mb"] else "   N/A"
        print(f"  {r['model_name']:<26} {wer_s}  "
              f"{r['wer_p50']:>5.1f}%  {r['wer_p90']:>5.1f}%  "
              f"{exact_s:>12}  {r['avg_rtf']:>6.3f}  {vram_s}")

    print("=" * W)
    print(f"\n  RTF < 1.0 = リアルタイム以上  WER p50/p90 = 中央値/上位10%閾値\n")

    # ─── 速度ランキング ──────────────────────────────────────────

    print("速度ランキング (avg_rtf 昇順):")
    print("-" * W)
    for rank, r in enumerate(sorted(rows, key=lambda x: x["avg_rtf"]), 1):
        wer_s = f"WER={r['avg_wer_pct']:.1f}%" if r["avg_wer_pct"] is not None else "WER=N/A"
        print(f"  #{rank:<2} {r['model_name']:<26}  RTF={r['avg_rtf']:.3f}  "
              f"avg={r['avg_elapsed_s']:.2f}s  {wer_s}")
    print()

    # ─── 総合スコア（精度 × 速度） ──────────────────────────────

    scoreable = [r for r in rows if r["avg_wer_pct"] is not None and r["avg_rtf"] > 0]
    if scoreable:
        print("総合スコア  (1 - WER) / RTF  ← 速くて正確なほど高い:")
        print("-" * W)
        for r in sorted(scoreable,
                         key=lambda x: (1 - x["avg_wer_pct"] / 100) / x["avg_rtf"],
                         reverse=True):
            score = (1 - r["avg_wer_pct"] / 100) / r["avg_rtf"]
            print(f"  {r['model_name']:<26}  score={score:6.1f}  "
                  f"(WER={r['avg_wer_pct']:.1f}%  RTF={r['avg_rtf']:.3f})")
        best = max(scoreable, key=lambda x: (1 - x["avg_wer_pct"] / 100) / x["avg_rtf"])
        print(f"\n  ★ 推薦モデル: {best['model_name']}  "
              f"(WER={best['avg_wer_pct']:.1f}%  RTF={best['avg_rtf']:.3f})\n")

    # ─── セッション一覧 ──────────────────────────────────────────

    print("セッション一覧:")
    print("-" * W)
    for s in session_meta:
        print(f"  {s['timestamp']}  n={s['n_samples']}  "
              f"device={s['device']}  compute={s['compute_type']}")
    print()

    # ─── JSON 保存 ────────────────────────────────────────────────

    ts_out   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{ts_out}_combined_report.json"
    out_data = {
        "generated_at": ts_out,
        "sessions":     [s["timestamp"] for s in session_meta],
        "n_sessions":   len(session_meta),
        "models":       rows,
    }
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
    print(f"Saved → {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
