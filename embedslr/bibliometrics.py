"""
Bibliometric indicators for EmbedSLR
====================================

• complete set of 10 (+1) indicators A … I  
• each indicator as a separate function  
• ability to jointly/group calculate the full report
"""

from __future__ import annotations

import itertools as it
from collections import Counter
from typing import Dict, List, Set, Callable, Any

import pandas as pd

# ────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────
def _kw_sets(series: pd.Series) -> List[Set[str]]:
    """Converts the *Author Keywords* column into a list of sets (lower-case)."""
    return [
        {w.strip().lower() for w in str(x).split(";") if w.strip()}
        for x in series.fillna("")
    ]


def _cited_sets(df: pd.DataFrame) -> List[Set[int]]:
    """
    For each article, returns a set of indexes of *other* articles from the dataset
    whose title appears in its references.
    """
    if {"Title", "Parsed_References"} - set(df.columns):
        # missing required columns
        return [set() for _ in range(len(df))]

    titles = df["Title"].fillna("").str.lower().str.strip().tolist()
    refs   = df["Parsed_References"].tolist()          # list of sets of strings

    cited: List[Set[int]] = []
    for i, ref_set in enumerate(refs):
        cited_i: Set[int] = set()
        for ref_str in ref_set:
            ref_low = str(ref_str).lower()
            for j, t in enumerate(titles):
                if i == j or not t:
                    continue
                if t in ref_low:
                    cited_i.add(j)
        cited.append(cited_i)
    return cited


def _mutual_citation_stats(df: pd.DataFrame) -> tuple[float, int]:
    """
    Returns:
        • average number of shared cited articles per pair (H)
        • total number of *unique* articles from the dataset
          that were cited at least once (I)
    """
    cited_sets = _cited_sets(df)
    n          = len(cited_sets)
    pairs      = n * (n - 1) / 2 or 1

    total_intersections = 0
    all_cited: Set[int] = set()

    for i, j in it.combinations(range(n), 2):
        inter = cited_sets[i] & cited_sets[j]
        total_intersections += len(inter)

    for s in cited_sets:
        all_cited.update(s)

    avg_per_pair = total_intersections / pairs
    total_unique = len(all_cited)
    return avg_per_pair, total_unique


# ────────────────────────────────────────────────────────────
# Preparing shared statistics (single pass through the data)
# ────────────────────────────────────────────────────────────
def _prepare_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a dictionary with a complete set of aggregates needed to calculate indicators.
    Called only once during bulk calculation, then passed
    to individual indicator functions.
    """
    refs = (
        df["Parsed_References"].tolist()
        if "Parsed_References" in df.columns
        else [set()] * len(df)
    )
    kws = _kw_sets(df.get("Author Keywords", pd.Series([""] * len(df))))
    n       = len(df)
    pairs   = n * (n - 1) / 2 or 1

    # aggregates
    tot_r_int = tot_r_jac = pairs_with_ref = 0
    uniq_refs: Set[str] = set()
    tot_k_int = tot_k_jac = pairs_with_kw = 0
    kw_cnt: Counter[str] = Counter()

    for i, j in it.combinations(range(n), 2):
        inter_r = refs[i] & refs[j]
        union_r = refs[i] | refs[j]
        inter_k = kws[i] & kws[j]
        union_k = kws[i] | kws[j]

        # references
        tot_r_int += len(inter_r)
        tot_r_jac += len(inter_r) / len(union_r) if union_r else 0.0
        if inter_r:
            pairs_with_ref += 1
            uniq_refs.update(inter_r)

        # keywords
        tot_k_int += len(inter_k)
        tot_k_jac += len(inter_k) / len(union_k) if union_k else 0.0
        if inter_k:
            pairs_with_kw += 1

    for kw_set in kws:
        kw_cnt.update(kw_set)

    # mutual citations
    avg_mut_cit, tot_mut_cit = _mutual_citation_stats(df)

    return dict(
        refs=refs,
        kws=kws,
        n=n,
        pairs=pairs,
        tot_r_int=tot_r_int,
        tot_r_jac=tot_r_jac,
        pairs_with_ref=pairs_with_ref,
        uniq_refs=uniq_refs,
        tot_k_int=tot_k_int,
        tot_k_jac=tot_k_jac,
        pairs_with_kw=pairs_with_kw,
        kw_cnt=kw_cnt,
        avg_mutual_cit=avg_mut_cit,
        total_mutual_cit=tot_mut_cit,
    )


# ────────────────────────────────────────────────────────────
# Functions for individual indicators
# ────────────────────────────────────────────────────────────
def indicator_a(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> float:
    """A – average number of shared references per article pair."""
    s = _stats or _prepare_stats(df)
    return s["tot_r_int"] / s["pairs"]


def indicator_a_prime(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> float:
    """A′ – average Jaccard (references) for all pairs."""
    s = _stats or _prepare_stats(df)
    return s["tot_r_jac"] / s["pairs"]


def indicator_b(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> float:
    """B – average number of shared keywords per pair."""
    s = _stats or _prepare_stats(df)
    return s["tot_k_int"] / s["pairs"]


def indicator_b_prime(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> float:
    """B′ – average Jaccard (keywords) for all pairs."""
    s = _stats or _prepare_stats(df)
    return s["tot_k_jac"] / s["pairs"]


def indicator_c(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> int:
    """C – number of pairs with at least one shared reference."""
    s = _stats or _prepare_stats(df)
    return s["pairs_with_ref"]


def indicator_d(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> int:
    """D – number of unique references shared by ≥2 articles."""
    s = _stats or _prepare_stats(df)
    return len(s["uniq_refs"])


def indicator_e(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> int:
    """E – total number of intersections (references) for all pairs."""
    s = _stats or _prepare_stats(df)
    return s["tot_r_int"]


def indicator_f(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> int:
    """F – number of pairs with ≥1 shared keyword."""
    s = _stats or _prepare_stats(df)
    return s["pairs_with_kw"]


def indicator_g(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> int:
    """G – number of keywords appearing in ≥2 articles."""
    s = _stats or _prepare_stats(df)
    return sum(c >= 2 for c in s["kw_cnt"].values())


def indicator_h(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> float:
    """H – average number of jointly cited articles per pair."""
    s = _stats or _prepare_stats(df)
    return s["avg_mutual_cit"]


def indicator_i(df: pd.DataFrame, *, _stats: Dict[str, Any] | None = None) -> int:
    """I – total number of *unique* mutually cited articles."""
    s = _stats or _prepare_stats(df)
    return s["total_mutual_cit"]


# map of names → functions (easy extension and iteration)
_INDICATOR_FUNCS: Dict[str, Callable[[pd.DataFrame, Any], float | int]] = {
    "A":  indicator_a,
    "A'": indicator_a_prime,
    "B":  indicator_b,
    "B'": indicator_b_prime,
    "C":  indicator_c,
    "D":  indicator_d,
    "E":  indicator_e,
    "F":  indicator_f,
    "G":  indicator_g,
    "H":  indicator_h,
    "I":  indicator_i,
}

# ────────────────────────────────────────────────────────────
# Main API
# ────────────────────────────────────────────────────────────
def indicators(df: pd.DataFrame) -> dict[str, float | int]:
    """
    Calculates the complete set of bibliometric indicators (A … I).

    Returns
    -------
    dict
        Keys:  A, A', B, B', C, D, E, F, G, H, I
    """
    stats = _prepare_stats(df)
    return {name: fn(df, _stats=stats) for name, fn in _INDICATOR_FUNCS.items()}


def full_report(
    df: pd.DataFrame,
    path: str | None = None,
    *,
    top_n: int | None = None,
) -> str:
    """
    Generates a formatted text report (and optionally saves it to *path*).
    """
    sub = df.head(top_n) if top_n else df
    ind = indicators(sub)

    order = ["A", "A'", "B", "B'", "C", "D", "E", "F", "G", "H", "I"]
    names = {
        "A":  "Avg shared refs / pair",
        "A'": "Jaccard refs",
        "B":  "Avg shared kws / pair",
        "B'": "Jaccard kws",
        "C":  "Pairs ≥1 common ref",
        "D":  "Unique common refs",
        "E":  "Sum intersections refs",
        "F":  "Pairs ≥1 common kw",
        "G":  "KWs in ≥2 articles",
        "H":  "Avg mutual citations",
        "I":  "Total mutual citations",
    }

    lines = ["==== BIBLIOMETRIC REPORT ===="]
    for k in order:
        v = ind[k]
        lines.append(f"{names[k]:32s}: {v:.4f}" if isinstance(v, float) else
                     f"{names[k]:32s}: {v}")

    report_txt = "\n".join(lines)

    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report_txt)

    return report_txt


# ── Direct execution (CLI test) ───────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("Usage: python bibliometric.py <scopus_export.csv> [topN]")
    csv = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else None
    df_  = pd.read_csv(csv)
    print(full_report(df_, top_n=topn))
