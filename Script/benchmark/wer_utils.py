"""
Text normalization and WER calculation for English ASR evaluation

Uses OpenAI Whisper's official EnglishTextNormalizer:
  - case-insensitive
  - removes punctuation and symbols
  - expands abbreviations (Mr. <-> Mister)
  - unifies number representations (ten <-> 10, twenties <-> 20s)
  - expands contractions (don't <-> do not)
Dependency: pip install whisper-normalizer
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
    """Normalize English text and return a list of words."""
    n = _get_normalizer()
    if n is not None:
        return n(text).split()
    # Fallback: simple normalization
    import re
    return re.sub(r"[^\w\s']", "", text.lower()).split()


def calc_wer(hyp: str, ref: str) -> float:
    """Word error rate = edit distance / number of reference words"""
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
    """Compare normalized word sequences and return the mismatched positions."""
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
            errors.append(f"  insert HYP[{i-1}]={hw[i-1]!r:14s}  (not in REF)")
            i -= 1
        else:
            errors.append(f"  delete REF[{j-1}]={rw[j-1]!r:14s}  (not in HYP)")
            j -= 1

    return list(reversed(errors))
