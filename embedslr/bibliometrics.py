from __future__ import annotations
import itertools as it
from collections import Counter
import pandas as pd


def _kw_sets(series: pd.Series) -> list[set[str]]:
    return [{w.strip().lower() for w in str(x).split(";") if w.strip()}
            for x in series.fillna("")]


def _avg_mutual_citations(df: pd.DataFrame) -> float:
    if {"Title", "Parsed_References"} - set(df.columns):
        return 0.0
    titles = df["Title"].str.lower().tolist()
    refs   = df["Parsed_References"].tolist()
    return sum(sum(t in r for t in titles) for r in refs) / len(refs)


def indicators(df: pd.DataFrame) -> dict[str, float | int]:
    refs = df["Parsed_References"].tolist() if "Parsed_References" in df.columns else [set()] * len(df)
    kws  = _kw_sets(df.get("Author Keywords", pd.Series([""]*len(df))))
    n    = len(df)
    pairs = n * (n - 1) / 2 or 1

    tot_r_int = tot_r_jac = p_r = 0
    uniq_r    = set()
    tot_k_int = tot_k_jac = p_k = 0
    kw_cnt    = Counter()

    for i, j in it.combinations(range(n), 2):
        ir = refs[i] & refs[j]; ur = refs[i] | refs[j]
        ik = kws[i]  & kws[j];  uk = kws[i] | kws[j]

        tot_r_int += len(ir)
        tot_r_jac += len(ir) / len(ur) if ur else 0
        if ir: p_r += 1; uniq_r.update(ir)

        tot_k_int += len(ik)
        tot_k_jac += len(ik) / len(uk) if uk else 0
        if ik: p_k += 1

    for s in kws:
        kw_cnt.update(s)

    return {
        "A":  tot_r_int / pairs,
        "A'": tot_r_jac / pairs,
        "B":  tot_k_int / pairs,
        "B'": tot_k_jac / pairs,
        "C":  p_r,
        "D":  len(uniq_r),
        "E":  tot_r_int,
        "F":  p_k,
        "G":  sum(c >= 2 for c in kw_cnt.values()),
        "H":  _avg_mutual_citations(df),
    }


def full_report(df: pd.DataFrame, path: str | None = None,
                *, top_n: int | None = None) -> str:
    sub = df.head(top_n) if top_n else df
    s = indicators(sub)
    order = ["A", "A'", "B", "B'", "C", "D", "E", "F", "G", "H"]
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
    }
    lines = ["==== BIBLIOMETRIC REPORT ===="]
    for k in order:
        v = s[k]
        lines.append(f"{names[k]:32s}: {v:.4f}" if isinstance(v, float) else
                     f"{names[k]:32s}: {v}")
    txt = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf‑8") as fh:
            fh.write(txt)
    return txt
