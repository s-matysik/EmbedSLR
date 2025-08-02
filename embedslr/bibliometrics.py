# embedslr/metrics.py
"""
Bibliometric indicators for EmbedSLR.
"""

from __future__ import annotations

import itertools as it
from collections import Counter

import pandas as pd


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _kw_sets(series: pd.Series) -> list[set[str]]:
    """Split Author Keywords; return list of lowercase sets."""
    return [
        {w.strip().lower() for w in str(x).split(";") if w.strip()}
        for x in series.fillna("")
    ]


def _mutual_citations_total(df: pd.DataFrame) -> int:
    """Return total number of times a dataset‑article cites another one."""
    if {"Title", "Parsed_References"} - set(df.columns):
        return 0
    titles = df["Title"].str.lower().tolist()
    refs = df["Parsed_References"].tolist()
    return sum(sum(t in r for t in titles) for r in refs)


def _avg_mutual_citations(df: pd.DataFrame) -> float:
    """Average mutual citations per article (indicator H)."""
    total = _mutual_citations_total(df)
    return total / len(df) if len(df) else 0.0


# ────────────────────────────────────────────────────────────
# Main API
# ────────────────────────────────────────────────────────────
def indicators(df: pd.DataFrame) -> dict[str, float | int]:
    """
    Compute ten bibliometric indicators (A … I).

    Returns
    -------
    dict
        Keys:
          A, A', B, B', C, D, E, F, G, H, I
    """
    refs = (
        df["Parsed_References"].tolist()
        if "Parsed_References" in df.columns
        else [set()] * len(df)
    )
    kws = _kw_sets(df.get("Author Keywords", pd.Series([""] * len(df))))
    n = len(df)
    pairs = n * (n - 1) / 2 or 1

    # aggregates
    tot_r_int = tot_r_jac = pairs_with_ref = 0
    uniq_refs: set[str] = set()
    tot_k_int = tot_k_jac = pairs_with_kw = 0
    kw_cnt: Counter[str] = Counter()

    for i, j in it.combinations(range(n), 2):
        inter_r = refs[i] & refs[j]
        union_r = refs[i] | refs[j]
        inter_k = kws[i] & kws[j]
        union_k = kws[i] | kws[j]

        # references
        tot_r_int += len(inter_r)
        tot_r_jac += len(inter_r) / len(union_r) if union_r else 0
        if inter_r:
            pairs_with_ref += 1
            uniq_refs.update(inter_r)

        # keywords
        tot_k_int += len(inter_k)
        tot_k_jac += len(inter_k) / len(union_k) if union_k else 0
        if inter_k:
            pairs_with_kw += 1

    for kw_set in kws:
        kw_cnt.update(kw_set)

    return {
        "A": tot_r_int / pairs,
        "A'": tot_r_jac / pairs,
        "B": tot_k_int / pairs,
        "B'": tot_k_jac / pairs,
        "C": pairs_with_ref,
        "D": len(uniq_refs),
        "E": tot_r_int,
        "F": pairs_with_kw,
        "G": sum(c >= 2 for c in kw_cnt.values()),
        "H": _avg_mutual_citations(df),
        "I": _mutual_citations_total(df),           # NEW – total mutual citations
    }


def full_report(
    df: pd.DataFrame,
    path: str | None = None,
    *,
    top_n: int | None = None,
) -> str:
    """
    Generate formatted text report and (optionally) write to *path*.
    """
    sub = df.head(top_n) if top_n else df
    ind = indicators(sub)

    order = ["A", "A'", "B", "B'", "C", "D", "E", "F", "G", "H", "I"]
    names = {
        "A": "Avg shared refs / pair",
        "A'": "Jaccard refs",
        "B": "Avg shared kws / pair",
        "B'": "Jaccard kws",
        "C": "Pairs ≥1 common ref",
        "D": "Unique common refs",
        "E": "Sum intersections refs",
        "F": "Pairs ≥1 common kw",
        "G": "KWs in ≥2 articles",
        "H": "Avg mutual citations",
        "I": "Total mutual citations",        # label for new metric
    }

    lines = ["==== BIBLIOMETRIC REPORT ===="]
    for k in order:
        v = ind[k]
        if isinstance(v, float):
            lines.append(f"{names[k]:32s}: {v:.4f}")
        else:
            lines.append(f"{names[k]:32s}: {v}")

    report_txt = "\n".join(lines)

    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report_txt)

    return report_txt


# Run as script for quick test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("Usage: python metrics.py <scopus_export.csv> [topN]")
    csv = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else None
    df_ = pd.read_csv(csv)
    print(full_report(df_, top_n=topn))
