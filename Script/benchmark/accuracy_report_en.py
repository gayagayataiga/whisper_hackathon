#!/usr/bin/env python3
"""
accuracy_report_en.py - English ASR benchmark detailed accuracy report

Loads the latest run (or a specified run) from results/en/ and displays
per-model WER details and the top error samples.

Usage:
    python Script/benchmark/accuracy_report_en.py           # latest run
    python Script/benchmark/accuracy_report_en.py --run 20260430_220714
    python Script/benchmark/accuracy_report_en.py --top 20  # top N error samples
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
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="English ASR detailed accuracy report")
    parser.add_argument("--run",   help="Timestamp to use (defaults to latest)")
    parser.add_argument("--top",   type=int, default=10, help="Show top N error samples (default: 10)")
    parser.add_argument("--model", nargs="+", dest="models", help="Restrict to specific models")
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print(f"[Error] {RESULTS_DIR} does not exist. Run whisper_benchmark.py first.")
        return

    # Determine timestamp
    if args.run:
        timestamp = args.run
    else:
        summaries = sorted(RESULTS_DIR.glob("*_summary.json"))
        if not summaries:
            print(f"[Error] No summary JSON found in {RESULTS_DIR}.")
            return
        timestamp = summaries[-1].stem.replace("_summary", "")

    print(f"Run: {timestamp}\n")

    # Get model list from summary
    summary_path = RESULTS_DIR / f"{timestamp}_summary.json"
    if not summary_path.exists():
        print(f"[Error] {summary_path} not found.")
        return
    summary = json.loads(summary_path.read_text())
    run_models = [m["model_name"] for m in summary["models"]]

    # Model filter
    target_models = args.models if args.models else MODEL_ORDER
    target_models = [m for m in target_models if m in run_models]
    if not target_models:
        target_models = run_models

    W = 90
    model_rows = []  # holds aggregated results after recalculation

    for model_name in target_models:
        safe_name = model_name.replace("/", "_").replace(".", "_")
        model_path = RESULTS_DIR / f"{timestamp}_{safe_name}.json"
        if not model_path.exists():
            print(f"[skip] {model_name}: {model_path.name} not found")
            continue

        data = json.loads(model_path.read_text())
        samples = data.get("samples", [])
        if not samples:
            continue

        # Recalculate WER from REF/HYP (may differ from normalization used at save time)
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
              f"exact={exact_n}/{len(wer_list)} ({exact_n/len(wer_list)*100:.1f}%)  "
              f"RTF={data.get('avg_rtf', 0):.3f}  "
              f"VRAM={data.get('avg_vram_mb', 0):.0f}MB")
        print("=" * W)

        errors = sorted(
            [s for s in samples if s["wer_recalc"] and s["wer_recalc"] > 0],
            key=lambda s: s["wer_recalc"],
            reverse=True,
        )

        print(f"\n  Errors: {len(errors)}/{len(samples)} samples"
              f"  (showing top {min(args.top, len(errors))})\n")

        for s in errors[: args.top]:
            print(f"  [{s['idx']:3d}]  WER={s['wer_recalc']*100:.1f}%  "
                  f"dur={s['duration_s']:.1f}s  RTF={s['rtf']:.3f}")
            print(f"    REF: {s['ref']}")
            print(f"    HYP: {s['hyp']}")
            for line in word_diff(s["hyp"], s["ref"]):
                print(f"    {line}")
            print()

        if not errors:
            print("  (all samples exact match)\n")

    # ─── Cross-model summary ─────────────────────────────────────
    print("=" * W)
    print("  SUMMARY  (sorted by WER asc)  *recalculated with whisper-normalizer")
    print("=" * W)
    print(f"  {'Model':<26} {'WER':>7}  {'Exact':>10}  {'RTF':>6}  {'VRAM':>7}")
    print("-" * W)

    model_rows.sort(key=lambda r: r["avg_wer"])
    for r in model_rows:
        wer_s    = f"{r['avg_wer']*100:>6.2f}%"
        exact_s  = f"{r['exact_n']}/{r['n_samples']} ({r['exact_n']/r['n_samples']*100:.0f}%)"
        vram_s   = f"{r['avg_vram_mb']:>5.0f}MB" if r["avg_vram_mb"] else "   N/A"
        print(f"  {r['model_name']:<26} {wer_s}  {exact_s:>12}  {r['avg_rtf']:>6.3f}  {vram_s}")

    print("=" * W)

    # Save JSON
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
