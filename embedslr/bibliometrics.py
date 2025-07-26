"""
Lightweight helpers for quick bibliometric diagnostics.
"""
from __future__ import annotations
import itertools as it
from collections import Counter
import pandas as pd


def _parse_set(series: pd.Series) -> list[set[str]]:
    res = []
    for cell in series.fillna(""):
        parts = {p.strip().lower() for p in str(cell).split(";") if p.strip()}
        res.append(parts)
    return res


def keyword_overlap(df: pd.DataFrame, col: str = "Author Keywords") -> dict[str, float]:
    sets = _parse_set(df[col])
    pairs = len(df) * (len(df) - 1) / 2
    inter = j_sum = 0
    for a, b in it.combinations(range(len(df)), 2):
        i = len(sets[a] & sets[b])
        u = len(sets[a] | sets[b])
        inter += i
        if u:
            j_sum += i / u
    return {
        "avg_common_keywords": inter / pairs if pairs else 0.0,
        "avg_jaccard_keywords": j_sum / pairs if pairs else 0.0,
        "keywords_>=2_articles": sum(v >= 2 for v in Counter(it.chain.from_iterable(sets)).values()),
    }


def full_report(df: pd.DataFrame, path: str | None = None) -> str:
    stats = keyword_overlap(df)
    lines = ["==== BIBLIOMETRIC REPORT ===="]
    for k, v in stats.items():
        lines.append(f"{k:30s}: {v:.4f}" if isinstance(v, float) else f"{k:30s}: {v}")
    txt = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf‑8") as fh:
            fh.write(txt)
    return txt
