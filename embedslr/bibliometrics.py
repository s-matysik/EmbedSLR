"""
Bibliometric indicators A–H
---------------------------
A   average shared references / pair
A’  Jaccard refs
B   average shared keywords / pair
B’  Jaccard keywords
C   pairs with ≥1 common ref
D   unique common refs
E   sum intersections refs
F   pairs with ≥1 common kw
G   keywords appearing ≥2 times
H   average mutual citations inside analysed set
"""
from __future__ import annotations
import itertools as it
from collections import Counter
import pandas as pd


def _kw_sets(series: pd.Series) -> list[set[str]]:
    return [{w.strip().lower() for w in str(x).split(";") if w.strip()}
            for x in series.fillna("")]


def _avg_mutual_citations(df: pd.DataFrame) -> float:
    if "Parsed_References" not in df.columns:
        return 0.0
    titles = df["Title"].str.lower().tolist()
    refs   = df["Parsed_References"].tolist()
    return sum(sum(t in r for t in titles) for r in refs) / len(refs)


def indicators(df: pd.DataFrame) -> dict[str, float | int]:
    refs = df["Parsed_References"].tolist()
    kws  = _kw_sets(df["Author Keywords"])
    n = len(df)
    pairs = n * (n - 1) / 2 or 1

    tot_ref_int = tot_ref_jac = pairs_ref = 0
    uniq_refs   = set()
    tot_kw_int  = tot_kw_jac = pairs_kw  = 0
    kw_counter  = Counter()

    for i, j in it.combinations(range(n), 2):
        ir = refs[i] & refs[j]; ur = refs[i] | refs[j]
        ik = kws[i]  & kws[j];  uk = kws[i] | kws[j]

        tot_ref_int += len(ir)
        tot_ref_jac += len(ir) / len(ur) if ur else 0
        if ir:
            pairs_ref += 1; uniq_refs.update(ir)

        tot_kw_int += len(ik)
        tot_kw_jac += len(ik) / len(uk) if uk else 0
        if ik:
            pairs_kw += 1

    for s in kws:
        kw_counter.update(s)

    return {
        "A_avg_shared_refs":       tot_ref_int / pairs,
        "A_prime_jaccard_refs":    tot_ref_jac / pairs,
        "B_avg_shared_kws":        tot_kw_int / pairs,
        "B_prime_jaccard_kws":     tot_kw_jac / pairs,
        "C_pairs_with_common_ref": pairs_ref,
        "D_unique_common_refs":    len(uniq_refs),
        "E_sum_intersections":     tot_ref_int,
        "F_pairs_with_common_kw":  pairs_kw,
        "G_kws_in_≥2_articles":    sum(c >= 2 for c in kw_counter.values()),
        "H_avg_mutual_citations":  _avg_mutual_citations(df),
    }


def full_report(df: pd.DataFrame, path: str | None = None,
                *, top_n: int | None = None) -> str:
    sub = df.head(top_n) if top_n else df
    stats = indicators(sub)

    lines = ["==== BIBLIOMETRIC REPORT ===="]
    order = [
        ("A) Avg shared refs / pair",  "A_avg_shared_refs"),
        ("A’) Jaccard refs",           "A_prime_jaccard_refs"),
        ("B) Avg shared kws / pair",   "B_avg_shared_kws"),
        ("B’) Jaccard kws",            "B_prime_jaccard_kws"),
        ("C) Pairs ≥1 common ref",     "C_pairs_with_common_ref"),
        ("D) Unique common refs",      "D_unique_common_refs"),
        ("E) Sum intersections refs",  "E_sum_intersections"),
        ("F) Pairs ≥1 common kw",      "F_pairs_with_common_kw"),
        ("G) KWs in ≥2 articles",      "G_kws_in_≥2_articles"),
        ("H) Avg mutual citations",    "H_avg_mutual_citations"),
    ]
    for lbl, k in order:
        val = stats[k]
        lines.append(f"{lbl:32s}: {val:.4f}" if isinstance(val, float)
                     else f"{lbl:32s}: {val}")

    txt = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf‑8") as fh:
            fh.write(txt)
    return txt
