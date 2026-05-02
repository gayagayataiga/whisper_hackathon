#!/usr/bin/env python3
"""
accuracy_report_en.py - 英語 ASR ベンチマーク 精度詳細レポート

results/en/ の最新ラン（または指定ラン）を読み込み、
モデル別の WER 詳細・誤り上位サンプルを表示する。

使い方:
    python Script/benchmark/accuracy_report_en.py           # 最新ラン
    python Script/benchmark/accuracy_report_en.py --run 20260430_220714
    python Script/benchmark/accuracy_report_en.py --top 20  # 誤り上位 N 件
    python Script/benchmark/accuracy_report_en.py --model tiny.en large-v3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wer_utils import calc_wer, word_diff

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "en"
MODEL_ORDER = [
    "tiny.en", "base.en", "small.en", "medium.en",
    "large-v2", "large-v3", "turbo",
    "distil-large-v3", "distil-large-v3.5",
]


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="英語 ASR 精度詳細レポート")
    parser.add_argument("--run",   help="タイムスタンプ指定（省略時は最新）")
    parser.add_argument("--top",   type=int, default=10, help="誤り上位 N サンプルを表示（default: 10）")
    parser.add_argument("--model", nargs="+", dest="models", help="対象モデルを絞る")
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print(f"[Error] {RESULTS_DIR} が存在しません。先に whisper_benchmark.py を実行してください。")
        return

    # タイムスタンプ決定
    if args.run:
        timestamp = args.run
    else:
        summaries = sorted(RESULTS_DIR.glob("*_summary.json"))
        if not summaries:
            print(f"[Error] {RESULTS_DIR} に summary JSON がありません。")
            return
        timestamp = summaries[-1].stem.replace("_summary", "")

    print(f"Run: {timestamp}\n")

    # summary からモデル一覧取得
    summary_path = RESULTS_DIR / f"{timestamp}_summary.json"
    if not summary_path.exists():
        print(f"[Error] {summary_path} が見つかりません。")
        return
    summary = json.loads(summary_path.read_text())
    run_models = [m["model_name"] for m in summary["models"]]

    # モデルフィルタ
    target_models = args.models if args.models else MODEL_ORDER
    target_models = [m for m in target_models if m in run_models]
    if not target_models:
        target_models = run_models

    W = 90
    model_rows = []  # 再計算後の集計を保持

    for model_name in target_models:
        safe_name = model_name.replace("/", "_").replace(".", "_")
        model_path = RESULTS_DIR / f"{timestamp}_{safe_name}.json"
        if not model_path.exists():
            print(f"[skip] {model_name}: {model_path.name} が見つかりません")
            continue

        data = json.loads(model_path.read_text())
        samples = data.get("samples", [])
        if not samples:
            continue

        # REF/HYP から WER を再計算（保存時の正規化と異なる可能性があるため）
        for s in samples:
            if s.get("ref") and s.get("hyp") is not None:
                s["wer_recalc"] = calc_wer(s["hyp"], s["ref"])
            else:
                s["wer_recalc"] = None

        wer_list = [s["wer_recalc"] for s in samples if s["wer_recalc"] is not None]
        avg_wer  = sum(wer_list) / len(wer_list) if wer_list else 0.0
        exact_n  = sum(1 for w in wer_list if w == 0.0)

        model_rows.append({
            "model_name":  model_name,
            "avg_wer":     avg_wer,
            "exact_n":     exact_n,
            "n_samples":   len(wer_list),
            "avg_rtf":     data.get("avg_rtf", 0),
            "avg_vram_mb": data.get("avg_vram_mb", 0),
        })

        print("=" * W)
        print(f"  {model_name}  ({data.get('description', '')})")
        print(f"  WER: avg={avg_wer*100:.2f}%  "
              f"完全一致={exact_n}/{len(wer_list)} ({exact_n/len(wer_list)*100:.1f}%)  "
              f"RTF={data.get('avg_rtf', 0):.3f}  "
              f"VRAM={data.get('avg_vram_mb', 0):.0f}MB")
        print("=" * W)

        errors = sorted(
            [s for s in samples if s["wer_recalc"] and s["wer_recalc"] > 0],
            key=lambda s: s["wer_recalc"],
            reverse=True,
        )

        print(f"\n  誤りあり: {len(errors)}/{len(samples)} サンプル"
              f"  （上位 {min(args.top, len(errors))} 件を表示）\n")

        for s in errors[: args.top]:
            print(f"  [{s['idx']:3d}]  WER={s['wer_recalc']*100:.1f}%  "
                  f"dur={s['duration_s']:.1f}s  RTF={s['rtf']:.3f}")
            print(f"    REF: {s['ref']}")
            print(f"    HYP: {s['hyp']}")
            for line in word_diff(s["hyp"], s["ref"]):
                print(f"    {line}")
            print()

        if not errors:
            print("  (全サンプル完全一致)\n")

    # ─── モデル横断サマリー ──────────────────────────────────────
    print("=" * W)
    print("  SUMMARY  (WER昇順)  ※whisper-normalizer で再計算")
    print("=" * W)
    print(f"  {'Model':<26} {'WER':>7}  {'完全一致':>10}  {'RTF':>6}  {'VRAM':>7}")
    print("-" * W)

    model_rows.sort(key=lambda r: r["avg_wer"])
    for r in model_rows:
        wer_s    = f"{r['avg_wer']*100:>6.2f}%"
        exact_s  = f"{r['exact_n']}/{r['n_samples']} ({r['exact_n']/r['n_samples']*100:.0f}%)"
        vram_s   = f"{r['avg_vram_mb']:>5.0f}MB" if r["avg_vram_mb"] else "   N/A"
        print(f"  {r['model_name']:<26} {wer_s}  {exact_s:>12}  {r['avg_rtf']:>6.3f}  {vram_s}")

    print("=" * W)

    # JSON 保存
    out_path = RESULTS_DIR / f"{timestamp}_accuracy_report.json"
    out_data = {
        "run":       timestamp,
        "dataset":   summary.get("dataset", ""),
        "n_samples": summary.get("n_samples", 0),
        "normalizer": "whisper-normalizer",
        "models":    [
            {
                "model_name":  r["model_name"],
                "avg_wer_pct": round(r["avg_wer"] * 100, 2),
                "exact_n":     r["exact_n"],
                "n_samples":   r["n_samples"],
                "avg_rtf":     round(r["avg_rtf"], 4),
                "avg_vram_mb": round(r["avg_vram_mb"], 1),
            }
            for r in model_rows
        ],
    }
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
    print(f"\nSaved → {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
