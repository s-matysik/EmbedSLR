"""
Bibliometric indicators for EmbedSLR
====================================
• komplet 10 wskaźników 
• działa zarówno dla pełnego zbioru, jak i podzbioru *top‑N* publikacji
"""

from __future__ import annotations

import itertools as it
from collections import Counter
from typing import List, Set

import pandas as pd


# ────────────────────────────────────────────────────────────
# Pomocnicze funkcje
# ────────────────────────────────────────────────────────────
def _kw_sets(series: pd.Series) -> List[Set[str]]:
    """Zamienia kolumnę *Author Keywords* na listę zbiorów (lower‑case)."""
    return [
        {w.strip().lower() for w in str(x).split(";") if w.strip()}
        for x in series.fillna("")
    ]


def _cited_sets(df: pd.DataFrame) -> List[Set[int]]:
    """
    Dla każdego artykułu zwraca zbiór indeksów *innych* artykułów z datasetu,
    których tytuł występuje w jego referencjach.
    """
    if {"Title", "Parsed_References"} - set(df.columns):
        # brak wymaganych kolumn
        return [set() for _ in range(len(df))]

    titles = df["Title"].fillna("").str.lower().str.strip().tolist()
    refs   = df["Parsed_References"].tolist()          # lista setów/zbiorów stringów

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
    Zwraca:
        • średnią liczbę wspólnych cytowanych artykułów na parę (H)
        • łączną liczbę *unikatowych* artykułów z datasetu,
          które zostały choć raz zacytowane (I)
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
# Główne API
# ────────────────────────────────────────────────────────────
def indicators(df: pd.DataFrame) -> dict[str, float | int]:
    """
    Oblicza 10 wskaźników bibliometrycznych (A … I).

    Zwraca
    -------
    dict
        Klucze:  A, A', B, B', C, D, E, F, G, H, I
    """
    # ── referencje i słowa kluczowe ─────────────────────────
    refs = (
        df["Parsed_References"].tolist()
        if "Parsed_References" in df.columns
        else [set()] * len(df)
    )
    kws = _kw_sets(df.get("Author Keywords", pd.Series([""] * len(df))))
    n       = len(df)
    pairs   = n * (n - 1) / 2 or 1

    # agregaty
    tot_r_int = tot_r_jac = pairs_with_ref = 0
    uniq_refs: Set[str] = set()
    tot_k_int = tot_k_jac = pairs_with_kw = 0
    kw_cnt: Counter[str] = Counter()

    for i, j in it.combinations(range(n), 2):
        inter_r = refs[i] & refs[j]
        union_r = refs[i] | refs[j]
        inter_k = kws[i] & kws[j]
        union_k = kws[i] | kws[j]

        # referencje
        tot_r_int += len(inter_r)
        tot_r_jac += len(inter_r) / len(union_r) if union_r else 0.0
        if inter_r:
            pairs_with_ref += 1
            uniq_refs.update(inter_r)

        # słowa kluczowe
        tot_k_int += len(inter_k)
        tot_k_jac += len(inter_k) / len(union_k) if union_k else 0.0
        if inter_k:
            pairs_with_kw += 1

    for kw_set in kws:
        kw_cnt.update(kw_set)

    # ── cytowania wzajemne (H, I) ───────────────────────────
    avg_mutual_cit, total_mutual_cit = _mutual_citation_stats(df)

    return {
        "A":  tot_r_int / pairs,           # average shared references per pair
        "A'": tot_r_jac / pairs,           # Jaccard of references
        "B":  tot_k_int / pairs,           # average shared keywords per pair
        "B'": tot_k_jac / pairs,           # Jaccard of keywords
        "C":  pairs_with_ref,              # pairs with at least one common ref
        "D":  len(uniq_refs),              # unique common references
        "E":  tot_r_int,                   # sum of intersections (refs)
        "F":  pairs_with_kw,               # pairs with at least one common kw
        "G":  sum(c >= 2 for c in kw_cnt.values()),  # keywords appearing ≥2 times
        "H":  avg_mutual_cit,              # *average* mutual citations per pair
        "I":  total_mutual_cit,            # *total unique* mutual citations
    }


def full_report(
    df: pd.DataFrame,
    path: str | None = None,
    *,
    top_n: int | None = None,
) -> str:
    """
    Generuje sformatowany raport tekstowy (i opcjonalnie zapisuje do *path*).
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


# ── Uruchomienie bezpośrednie (test CLI) ───────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("Usage: python metrics.py <scopus_export.csv> [topN]")
    csv = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else None
    df_  = pd.read_csv(csv)
    print(full_report(df_, top_n=topn))
