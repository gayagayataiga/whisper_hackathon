"""
英語 ASR 評価用のテキスト正規化と WER 計算

OpenAI Whisper 公式の EnglishTextNormalizer を使用：
  - 大文字小文字無視
  - 句読点・記号削除
  - 略語展開 (Mr. ↔ Mister)
  - 数字表記統一 (ten ↔ 10, twenties ↔ 20s)
  - 縮約形展開 (don't ↔ do not)
依存: pip install whisper-normalizer
"""

from typing import Optional

_normalizer: Optional[object] = None
_normalizer_init = False


def _get_normalizer():
    global _normalizer, _normalizer_init
    if not _normalizer_init:
        _normalizer_init = True
        try:
            from whisper_normalizer.english import EnglishTextNormalizer
            _normalizer = EnglishTextNormalizer()
        except ImportError:
            _normalizer = None
    return _normalizer


def normalize(text: str) -> list[str]:
    """英語テキストを正規化して単語リストを返す"""
    n = _get_normalizer()
    if n is not None:
        return n(text).split()
    # フォールバック: 単純正規化
    import re
    return re.sub(r"[^\w\s']", "", text.lower()).split()


def calc_wer(hyp: str, ref: str) -> float:
    """単語誤り率 = 編集距離 / 参照語数"""
    hw, rw = normalize(hyp), normalize(ref)
    if not rw:
        return 0.0
    d = list(range(len(rw) + 1))
    for h in hw:
        nd = [d[0] + 1]
        for j, r in enumerate(rw, 1):
            nd.append(min(d[j] + 1, nd[-1] + 1, d[j - 1] + (0 if h == r else 1)))
        d = nd
    return d[-1] / len(rw)


def word_diff(hyp: str, ref: str) -> list[str]:
    """正規化後の単語列を比較し、不一致箇所を返す"""
    hw = normalize(hyp)
    rw = normalize(ref)

    n, m = len(hw), len(rw)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (0 if hw[i - 1] == rw[j - 1] else 1),
            )

    errors = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if hw[i - 1] == rw[j - 1] else 1):
            if hw[i - 1] != rw[j - 1]:
                errors.append(f"  subst  REF[{j-1}]={rw[j-1]!r:14s}  HYP={hw[i-1]!r}")
            i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            errors.append(f"  insert HYP[{i-1}]={hw[i-1]!r:14s}  (REF にない)")
            i -= 1
        else:
            errors.append(f"  delete REF[{j-1}]={rw[j-1]!r:14s}  (HYP にない)")
            j -= 1

    return list(reversed(errors))
