#!/usr/bin/env python3
"""
accuracy_report.py - large-v3 を正解としてモデル別の正答率・CER を集計する。

正答率  : 10 run 中、large-v3 の最頻出テキストと完全一致した割合
CER     : Character Error Rate = edit_distance / len(reference)  (小さいほど良い)
文字精度: 1 - CER
"""

import json
from pathlib import Path


# ============================================================
# Levenshtein 編集距離（外部ライブラリ不要）
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
                prev[j] + 1,       # 削除
                curr[j - 1] + 1,   # 挿入
                prev[j - 1] + (0 if ca == cb else 1),  # 置換
            ))
        prev = curr
    return prev[-1]


# ============================================================
# 集計
# ============================================================

RESULTS_DIR  = Path(__file__).parent.parent.parent / "results" / "stress"
MODEL_ORDER  = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

# 最新の summary ファイルを自動選択
summary_files = sorted(RESULTS_DIR.glob("*_summary.json"))
if not summary_files:
    raise FileNotFoundError(f"No summary JSON found in {RESULTS_DIR}")
summary_path = summary_files[-1]
timestamp    = summary_path.stem.replace("_summary", "")

print(f"Using: {summary_path.name}\n")

summary = json.loads(summary_path.read_text())

# large-v3 の最頻出テキストを正解とする
lv3_texts = [r["text"] for r in summary["large-v3"]["runs"]]
from collections import Counter
reference = Counter(lv3_texts).most_common(1)[0][0]
print(f"Reference (large-v3 mode): {reference!r}\n")

# ============================================================
# 各モデルの計算
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

    # 最頻出出力と、そのときの差分
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
# 表示
# ============================================================

W = 70
print("=" * W)
print(f"  {'Model':<12} {'正答率':>8} {'文字精度':>8} {'CER':>7}  出力テキスト")
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
print("誤りの詳細:")
print("-" * W)
for r in rows:
    if r["mode_dist"] > 0:
        print(f"  {r['model']:<12}  edit_dist={r['mode_dist']}  CER={r['mode_cer']*100:.1f}%")
        print(f"    正解: {reference!r}")
        print(f"    出力: {r['mode_text']!r}")
        # 文字単位で差分をハイライト
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
# JSON 保存
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
