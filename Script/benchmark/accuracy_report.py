#!/usr/bin/env python3
"""
accuracy_report.py - Aggregate per-model accuracy rate and CER using large-v3 as the reference.

Accuracy rate : fraction of runs (out of 10) that exactly match the most frequent large-v3 output
CER           : Character Error Rate = edit_distance / len(reference)  (lower is better)
Char accuracy : 1 - CER
"""

import json
from pathlib import Path


# ============================================================
# Levenshtein edit distance (no external library needed)
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
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + (0 if ca == cb else 1),  # substitution
            ))
        prev = curr
    return prev[-1]


# ============================================================
# Aggregation
# ============================================================

RESULTS_DIR  = Path(__file__).parent.parent.parent / "results" / "stress"
MODEL_ORDER  = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

# Automatically select the most recent summary file
summary_files = sorted(RESULTS_DIR.glob("*_summary.json"))
if not summary_files:
    raise FileNotFoundError(f"No summary JSON found in {RESULTS_DIR}")
summary_path = summary_files[-1]
timestamp    = summary_path.stem.replace("_summary", "")

print(f"Using: {summary_path.name}\n")

summary = json.loads(summary_path.read_text())

# Use the most frequent large-v3 output as the reference
lv3_texts = [r["text"] for r in summary["large-v3"]["runs"]]
from collections import Counter
reference = Counter(lv3_texts).most_common(1)[0][0]
print(f"Reference (large-v3 mode): {reference!r}\n")

# ============================================================
# Per-model computation
# ============================================================

rows: list[dict] = []

for model_size in MODEL_ORDER:
    data   = summary[model_size]
    texts  = [r["text"] for r in data["runs"]]
    n      = len(texts)

    exact_matches = sum(1 for t in texts if t == reference)
    exact_rate    = exact_matches / n

    distances = [levenshtein(t, reference) for t in texts]
    avg_cer   = sum(d / max(len(reference), 1) for d in distances) / n
    char_acc  = 1.0 - avg_cer

    # Most frequent output and its edit distance from the reference
    mode_text = Counter(texts).most_common(1)[0][0]
    mode_dist = levenshtein(mode_text, reference)
    mode_cer  = mode_dist / max(len(reference), 1)

    rows.append({
        "model":        model_size,
        "exact_rate":   exact_rate,
        "char_acc":     char_acc,
        "avg_cer":      avg_cer,
        "mode_text":    mode_text,
        "mode_cer":     mode_cer,
        "mode_dist":    mode_dist,
        "n_exact":      exact_matches,
        "n_runs":       n,
    })

# ============================================================
# Display
# ============================================================

W = 70
print("=" * W)
print(f"  {'Model':<12} {'Accuracy':>8} {'CharAcc':>8} {'CER':>7}  Output text")
print("-" * W)
for r in rows:
    marker = "✓" if r["exact_rate"] == 1.0 else ("△" if r["exact_rate"] > 0 else "✗")
    print(
        f"  {r['model']:<12} "
        f"{r['exact_rate']*100:>6.1f}%  "
        f"{r['char_acc']*100:>6.1f}%  "
        f"{r['avg_cer']*100:>5.1f}%  "
        f"{marker} {r['mode_text']!r}"
    )
print("=" * W)

print()
print("Error details:")
print("-" * W)
for r in rows:
    if r["mode_dist"] > 0:
        print(f"  {r['model']:<12}  edit_dist={r['mode_dist']}  CER={r['mode_cer']*100:.1f}%")
        print(f"    REF: {reference!r}")
        print(f"    HYP: {r['mode_text']!r}")
        # Highlight differences at the character level
        diff = []
        for i, (c_ref, c_hyp) in enumerate(
            zip(reference.ljust(max(len(reference), len(r['mode_text']))),
                r['mode_text'].ljust(max(len(reference), len(r['mode_text']))))
        ):
            if c_ref != c_hyp:
                diff.append(f"    pos {i}: {c_ref!r} → {c_hyp!r}")
        for d in diff:
            print(d)
        print()

# ============================================================
# JSON save
# ============================================================

out = {
    "timestamp":  timestamp,
    "reference":  reference,
    "reference_model": "large-v3",
    "models": rows,
}

out_path = RESULTS_DIR / f"{timestamp}_accuracy.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"Saved → {out_path}")
