"""
Compute bibliometric indicators A–H on a (sub)set of publications.

Indicators (as requested):
 A  – average number of shared references per pair
 A’ – Jaccard index for references
 B  – average number of shared keywords per pair
 B’ – Jaccard index for keywords
 C  – number of pairs with ≥1 common reference
 D  – number of unique common references
 E  – sum of all intersections (refs)
 F  – number of pairs with ≥1 common keyword
 G  – number of keywords appearing ≥2 times
 H  – average mutual citations within the analysed set
"""
from __future__ import annotations
import itertools as it
from collections import Counter
import pandas as pd


def _kw_sets(series: pd.Series) -> list[set[str]]:
    return [
        {p.strip().lower() for p in str(cell).split(";") if p.strip()}
        for cell in series.fillna("")
    ]


def _avg_mutual_citations(df: pd.DataFrame) -> float:
    if "Parsed_References" not in df.columns:
        return 0.0
    titles = df["Title"].str.lower().tolist()
    refs   = df["Parsed_References"].tolist()
    cnt, total = 0, 0
    for rset in refs:
        total += sum(t in rset for t in titles)
        cnt   += 1
    return total / cnt if cnt else 0.0


def indicators(df: pd.DataFrame) -> dict[str, float | int]:
    refs = df["Parsed_References"].tolist()
    kws  = _kw_sets(df["Author Keywords"])
    n    = len(df)
    pairs = n * (n - 1) / 2

    tot_ref_inter = tot_ref_jacc = pairs_ref_common = 0
    uniq_ref_inter = set()

    tot_kw_inter = tot_kw_jacc = pairs_kw_common = 0
    kw_counter = Counter()

    for i, j in it.combinations(range(n), 2):
        # references
        inter_r = refs[i] & refs[j]
        union_r = refs[i] | refs[j]
        if inter_r:
            pairs_ref_common += 1
            uniq_ref_inter.update(inter_r)
        tot_ref_inter += len(inter_r)
        if union_r:
            tot_ref_jacc += len(inter_r) / len(union_r)

        # keywords
        inter_k = kws[i] & kws[j]
        union_k = kws[i] | kws[j]
        if inter_k:
            pairs_kw_common += 1
        tot_kw_inter += len(inter_k)
        if union_k:
            tot_kw_jacc += len(inter_k) / len(union_k)

    for s in kws:
        kw_counter.update(s)

    return {
        "A_avg_shared_refs":          tot_ref_inter / pairs if pairs else 0,
        "A_prime_jaccard_refs":       tot_ref_jacc / pairs if pairs else 0,
        "B_avg_shared_kws":           tot_kw_inter / pairs if pairs else 0,
        "B_prime_jaccard_kws":        tot_kw_jacc / pairs if pairs else 0,
        "C_pairs_with_common_ref":    pairs_ref_common,
        "D_unique_common_refs":       len(uniq_ref_inter),
        "E_sum_intersections_refs":   tot_ref_inter,
        "F_pairs_with_common_kw":     pairs_kw_common,
        "G_keywords_≥2_articles":     sum(c >= 2 for c in kw_counter.values()),
        "H_avg_mutual_citations":     _avg_mutual_citations(df),
    }


def full_report(df: pd.DataFrame, path: str | None = None, *,
                top_n: int | None = None) -> str:
    subset = df.head(top_n) if top_n else df
    stats = indicators(subset)

    order = [
        ("A) Average shared refs / pair",  "A_avg_shared_refs"),
        ("A’) Jaccard refs",               "A_prime_jaccard_refs"),
        ("B) Average shared kws / pair",   "B_avg_shared_kws"),
        ("B’) Jaccard kws",                "B_prime_jaccard_kws"),
        ("C) Pairs ≥1 common ref",         "C_pairs_with_common_ref"),
        ("D) Unique common refs",          "D_unique_common_refs"),
        ("E) Sum intersections refs",      "E_sum_intersections_refs"),
        ("F) Pairs ≥1 common kw",          "F_pairs_with_common_kw"),
        ("G) Keywords in ≥2 articles",     "G_keywords_≥2_articles"),
        ("H) Avg mutual citations",        "H_avg_mutual_citations"),
    ]
    lines = ["==== BIBLIOMETRIC REPORT ===="]
    for label, key in order:
        val = stats[key]
        s = f"{val:.4f}" if isinstance(val, float) else str(val)
        lines.append(f"{label:35s}: {s}")

    txt = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf‑8") as fh:
            fh.write(txt)
    return txt
