#!/usr/bin/env python3
"""
combined_report.py - 複数回ストレステストの結果を統合して正答率・速度を集計する。

全 summary JSON を自動検出し、全 run を結合して集計する。
"""

import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "stress"
MODEL_ORDER = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
AUDIO_DURATION_S = 3.19   # BASIC5000_0001.wav


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
# 全 summary を読み込んで run をモデル別に集約
# ============================================================

summary_files = sorted(RESULTS_DIR.glob("*_summary.json"))
if not summary_files:
    raise FileNotFoundError(f"No summary JSON found in {RESULTS_DIR}")

print(f"Found {len(summary_files)} summary file(s):")
for f in summary_files:
    print(f"  {f.name}")
print()

# model_size -> runs list
all_runs: dict[str, list[dict]] = {m: [] for m in MODEL_ORDER}

for sf in summary_files:
    data = json.loads(sf.read_text())
    for model_size in MODEL_ORDER:
        if model_size in data:
            all_runs[model_size].extend(data[model_size]["runs"])

# large-v3 の最頻出テキストを正解とする
lv3_texts = [r["text"] for r in all_runs["large-v3"]]
reference = Counter(lv3_texts).most_common(1)[0][0]
ref_len   = max(len(reference), 1)

print(f"Reference (large-v3 mode, n={len(lv3_texts)}): {reference!r}\n")

# ============================================================
# 集計
# ============================================================

rows: list[dict] = []

for model_size in MODEL_ORDER:
    runs = all_runs[model_size]
    n    = len(runs)

    texts  = [r["text"] for r in runs]
    times  = [r["inference_time_s"] for r in runs]
    vrams  = [r["vram_after_mb"] for r in runs]
    rtfs   = [t / AUDIO_DURATION_S for t in times]

    dists        = [levenshtein(t, reference) for t in texts]
    exact_n      = sum(1 for d in dists if d == 0)
    exact_rate   = exact_n / n
    avg_cer      = sum(d / ref_len for d in dists) / n
    char_acc     = 1.0 - avg_cer

    avg_time = sum(times) / n
    max_time = max(times)
    min_time = min(times)
    avg_rtf  = sum(rtfs)  / n
    avg_vram = sum(vrams) / n
    max_vram = max(vrams)

    mode_text = Counter(texts).most_common(1)[0][0]
    mode_dist = levenshtein(mode_text, reference)

    rows.append({
        "model":       model_size,
        "n_runs":      n,
        "exact_rate":  exact_rate,
        "char_acc":    char_acc,
        "avg_cer":     avg_cer,
        "avg_time_s":  avg_time,
        "max_time_s":  max_time,
        "min_time_s":  min_time,
        "avg_rtf":     avg_rtf,
        "avg_vram_mb": avg_vram,
        "max_vram_mb": max_vram,
        "mode_text":   mode_text,
        "mode_dist":   mode_dist,
    })

# ============================================================
# 表示
# ============================================================

W = 82
print("=" * W)
print(f"  Combined results: {len(summary_files)} sessions × {len(all_runs['large-v3'])} runs / model")
print("=" * W)
print(
    f"  {'Model':<12}  {'正答率':>6}  {'文字精度':>6}  {'CER':>5}  "
    f"{'avg_time':>8}  {'RTF':>5}  {'avg_VRAM':>9}"
)
print("-" * W)
for r in rows:
    ok = "✓" if r["exact_rate"] == 1.0 else ("△" if r["exact_rate"] > 0 else "✗")
    print(
        f"  {r['model']:<12}  "
        f"{r['exact_rate']*100:>5.1f}%  "
        f"{r['char_acc']*100:>5.1f}%  "
        f"{r['avg_cer']*100:>4.1f}%  "
        f"{r['avg_time_s']:>7.2f}s  "
        f"{r['avg_rtf']:>5.2f}  "
        f"{r['avg_vram_mb']:>7.0f}MB  "
        f"{ok}"
    )
print("=" * W)

print()
print(f"  RTF = inference_time / audio_duration ({AUDIO_DURATION_S}s).  RTF < 1.0 = リアルタイム以下")
print()

# ── 誤りの詳細 ──────────────────────────────────────────────
print("誤りの詳細:")
print("-" * W)
any_error = False
for r in rows:
    if r["mode_dist"] > 0:
        any_error = True
        print(f"  {r['model']:<12}  edit_dist={r['mode_dist']}  CER={r['avg_cer']*100:.1f}%")
        print(f"    正解: {reference!r}")
        print(f"    出力: {r['mode_text']!r}")
        ref_p = reference.ljust(max(len(reference), len(r["mode_text"])))
        hyp_p = r["mode_text"].ljust(max(len(reference), len(r["mode_text"])))
        for i, (c_ref, c_hyp) in enumerate(zip(ref_p, hyp_p)):
            if c_ref != c_hyp:
                c_ref_d = c_ref.strip() or "(なし)"
                c_hyp_d = c_hyp.strip() or "(なし)"
                print(f"    pos {i:2d}: {c_ref_d!r} → {c_hyp_d!r}")
        print()
if not any_error:
    print("  (誤りなし)")

# ── 速度ランキング ───────────────────────────────────────────
print("速度ランキング (avg_time 昇順):")
print("-" * W)
speed_ranked = sorted(rows, key=lambda x: x["avg_time_s"])
for rank, r in enumerate(speed_ranked, 1):
    acc_str = f"正答率 {r['exact_rate']*100:.0f}%  文字精度 {r['char_acc']*100:.1f}%"
    print(f"  #{rank}  {r['model']:<12}  {r['avg_time_s']:.2f}s (RTF={r['avg_rtf']:.2f})  {acc_str}")
print()

# ── 総合スコア（正確さ × 速度） ──────────────────────────────
# スコア = 文字精度 / avg_time_s  (高いほど速くて正確)
print("総合スコア (文字精度 / avg_time — 速さと精度のバランス):")
print("-" * W)
for r in sorted(rows, key=lambda x: x["char_acc"] / x["avg_time_s"], reverse=True):
    score = r["char_acc"] / r["avg_time_s"]
    print(f"  {r['model']:<12}  score={score:.2f}  (文字精度={r['char_acc']*100:.1f}%  avg={r['avg_time_s']:.2f}s)")
print()

# ============================================================
# JSON 保存
# ============================================================

from datetime import datetime
ts_out   = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = RESULTS_DIR / f"{ts_out}_combined_report.json"
out_data = {
    "sessions":    [f.name for f in summary_files],
    "n_sessions":  len(summary_files),
    "reference":   reference,
    "audio_duration_s": AUDIO_DURATION_S,
    "models":      rows,
}
out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
print(f"Saved → {out_path}")
