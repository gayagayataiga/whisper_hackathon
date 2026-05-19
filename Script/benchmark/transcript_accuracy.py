#!/usr/bin/env python3
"""
transcript_accuracy.py - Aggregate per-model accuracy rate and CER using all texts under music/ans/ as reference.

Format: STEM:text  (STEM = WAV filename without the .wav extension)

Usage:
    python Script/benchmark/transcript_accuracy.py                      # latest results/news/
    python Script/benchmark/transcript_accuracy.py --run 20260429_184650_news_bench
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path


REPO_ROOT   = Path(__file__).parent.parent.parent
ANS_DIR     = REPO_ROOT / "music" / "ans"
RESULTS_DIR = REPO_ROOT / "results" / "news"

MODEL_ORDER = ["tiny", "base", "small", "medium", "large-v2", "large-v3",
               "kotoba-whisper-v2"]


# ============================================================
# Normalization
# ============================================================

# Punctuation and symbols (Japanese comma/period, brackets, exclamation, etc.)
_PUNCT_RE = re.compile(
    r"[、。，．・！？…「」『』【】〔〕（）()!?\s　～〜~]"
)

# Single-digit kanji numerals -> Arabic digits (十/百/千/万 are context-dependent, so not converted)
_KANJI_DIGIT = str.maketrans("〇一二三四五六七八九", "0123456789")


def normalize(text: str) -> str:
    """Remove punctuation/spaces and unify fullwidth/kanji digits. Kanji characters are kept as-is."""
    text = unicodedata.normalize("NFKC", text)   # fullwidth alphanumerics -> halfwidth (５->5, etc.)
    text = text.translate(_KANJI_DIGIT)           # 一->1, 二->2 ... 九->9
    text = _PUNCT_RE.sub("", text)               # remove punctuation and spaces
    return text


# ============================================================
# Levenshtein edit distance
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
# Load all texts under ans/
# ============================================================

def load_all_ans(ans_dir: Path) -> dict[str, str]:
    """Return { stem: text } merged from all ans files."""
    ref: dict[str, str] = {}
    for txt in sorted(ans_dir.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            sep = line.index(":")
            key = line[:sep].strip()
            val = line[sep + 1:].strip()
            ref[key] = val
    return ref


# ============================================================
# Load benchmark result JSON files
# ============================================================

def load_model_files(run_prefix: str) -> dict[str, list[dict]]:
    model_results: dict[str, list[dict]] = {}
    for p in RESULTS_DIR.glob(f"{run_prefix}_*.json"):
        if "_summary" in p.name or "_accuracy" in p.name:
            continue
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        model_size = data.get("model_size",
                              p.stem.replace(f"{run_prefix}_", "").replace("_", "-"))
        model_results[model_size] = data.get("files", [])
    return model_results


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                        help="Timestamp prefix. Defaults to the latest run.")
    args = parser.parse_args()

    ref_map = load_all_ans(ANS_DIR)
    print(f"Reference texts: {len(ref_map)} entries from {ANS_DIR.name}/")

    if args.run:
        run_prefix = args.run
    else:
        summaries = sorted(RESULTS_DIR.glob("*_summary.json"))
        if not summaries:
            raise FileNotFoundError(f"No summary JSON in {RESULTS_DIR}")
        run_prefix = summaries[-1].stem.replace("_summary", "")

    print(f"Run: {run_prefix}\n")

    model_results = load_model_files(run_prefix)
    if not model_results:
        raise FileNotFoundError(f"No per-file JSON found for prefix: {run_prefix}")

    rows: list[dict] = []

    ordered = [m for m in MODEL_ORDER if m in model_results]
    ordered += [m for m in model_results if m not in ordered]

    for model_size in ordered:
        recs = model_results[model_size]
        cers: list[float] = []
        exact = 0
        missing = 0
        sample_errors: list[tuple[str, str, str]] = []

        times: list[float] = []
        vrams: list[float] = []

        for rec in recs:
            stem = Path(rec["file"]).stem
            ref_text = ref_map.get(stem)
            if ref_text is None:
                missing += 1
                continue

            hyp_text = rec["text"]
            ref_norm = normalize(ref_text)
            hyp_norm = normalize(hyp_text)
            dist    = levenshtein(hyp_norm, ref_norm)
            ref_len = max(len(ref_norm), 1)
            cer     = dist / ref_len
            cers.append(cer)

            if rec.get("inference_time_s") is not None:
                times.append(rec["inference_time_s"])
            if rec.get("vram_mb") is not None:
                vrams.append(rec["vram_mb"])

            if cer == 0.0:
                exact += 1
            elif len(sample_errors) < 3:
                sample_errors.append((stem, ref_text, hyp_text))

        n = len(cers)
        if n == 0:
            continue

        rows.append({
            "model":      model_size,
            "n":          n,
            "missing":    missing,
            "exact":      exact,
            "exact_rate": exact / n,
            "char_acc":   1.0 - sum(cers) / n,
            "avg_cer":    sum(cers) / n,
            "avg_time_s": sum(times) / len(times) if times else None,
            "max_vram_mb": max(vrams) if vrams else None,
            "sample_errors": sample_errors,
        })

    # ── Display ──────────────────────────────────────────────────
    W = 88
    print("=" * W)
    print(f"  Reference: all files in music/ans/  *punctuation and fullwidth/kanji digits are normalized before comparison")
    print(f"  {'Model':<22} {'Accuracy':>8} {'CharAcc':>8} {'CER':>7}  {'avg_time':>9}  {'max_VRAM':>9}  N")
    print("-" * W)
    for r in rows:
        marker = "✓" if r["exact_rate"] == 1.0 else ("△" if r["exact_rate"] > 0 else "✗")
        time_str = f"{r['avg_time_s']:>7.2f}s" if r["avg_time_s"] is not None else "      N/A"
        vram_str = f"{r['max_vram_mb']:>7.0f}MB" if r["max_vram_mb"] is not None else "      N/A"
        print(
            f"  {r['model']:<22} "
            f"{r['exact_rate']*100:>6.1f}%  "
            f"{r['char_acc']*100:>6.1f}%  "
            f"{r['avg_cer']*100:>5.1f}%  "
            f"{time_str}  {vram_str}  "
            f"{marker} {r['n']}"
        )
    print("=" * W)

    print("\nError samples (up to 3 per model):")
    print("-" * W)
    for r in rows:
        if not r["sample_errors"]:
            continue
        print(f"\n  [{r['model']}]")
        for stem, ref, hyp in r["sample_errors"]:
            print(f"    {stem}")
            print(f"      REF: {ref}")
            print(f"      HYP: {hyp}")

    # ── JSON save ────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"{run_prefix}_transcript_accuracy.json"
    out_data = {
        "run":       run_prefix,
        "reference": "music/ans/",
        "models": [
            {k: v for k, v in r.items() if k != "sample_errors"}
            for r in rows
        ],
    }
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
    print(f"\nSaved → {out_path.name}")


if __name__ == "__main__":
    main()
