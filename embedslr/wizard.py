#!/usr/bin/env python3
"""
Interactive terminal wizard for EmbedSLR
========================================
Run with::

    python -m embedslr.wizard

The script:

1.  Asks for a Scopus / WoS CSV export.
2.  Asks for the research query.
3.  Lets you choose the embedding provider & model (lists come
    directly from ``embeddings.list_models`` – always in sync).
4.  (Optional) lets you restrict the bibliometric calculations
    to the *N* most relevant papers.
5.  Generates the same 11‑indicator report that Colab shows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import get_embeddings, list_models
try:                                    # new name in recent versions
    from .metrics import full_report
except ImportError:                     # fallback for older clones
    from .bibliometric import full_report


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
_TITLE_CANDIDATES     = ("Article Title", "Title", "TI")
_ABSTRACT_CANDIDATES  = ("Abstract", "AB")


def _ask(msg: str, *, default: str | None = None) -> str:
    prompt = f"{msg} " + (f"[{default}] " if default else "")
    ans = input(prompt).strip()
    return ans or (default or "")


def _choose(options: list[str], msg: str, *, default: int = 0) -> str:
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        sel = _ask(msg, default=str(default + 1))
        if sel.isdigit() and 1 <= int(sel) <= len(options):
            return options[int(sel) - 1]
        print("Invalid selection. Try again.")


def _first_existing(cols: tuple[str, ...], df: pd.DataFrame) -> str | None:
    return next((c for c in cols if c in df.columns), None)


# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------
def main() -> None:
    print("=== EmbedSLR – terminal wizard ===")
    csv_path = Path(_ask("Path to the CSV export:")).expanduser()
    if not csv_path.is_file():
        sys.exit("File not found – aborting.")

    df = pd.read_csv(csv_path, low_memory=False)

    query = _ask("Research query:")

    # ------------------------------------------------------------------
    # Choose provider & model
    # ------------------------------------------------------------------
    providers = list(list_models().keys())
    print("\nChoose embedding provider:")
    provider = _choose(providers, "Provider:", default=providers.index("sbert"))

    models = list_models()[provider]
    print(f"\nModels for {provider}:")
    model = _choose(models, "Model:", default=0)

    # ------------------------------------------------------------------
    # Optional top‑N selection for metrics
    # ------------------------------------------------------------------
    top_n_str = _ask("\nTop‑N publications for metrics (ENTER = all):")
    top_n = int(top_n_str) if top_n_str.isdigit() else None

    # ------------------------------------------------------------------
    # Build combined text field
    # ------------------------------------------------------------------
    title_col    = _first_existing(_TITLE_CANDIDATES, df)
    abstract_col = _first_existing(_ABSTRACT_CANDIDATES, df)
    if not title_col:
        sys.exit("No title column found – aborting.")

    df["combined_text"] = (
        df[title_col].fillna("").astype(str)
        + " "
        + df.get(abstract_col, "").fillna("").astype(str)
    )

    # ------------------------------------------------------------------
    # Generate embeddings
    # ------------------------------------------------------------------
    print("\n⏳ Generating embeddings …")
    art_embs  = get_embeddings(df["combined_text"].tolist(),
                               provider=provider, model=model)
    query_emb = get_embeddings([query], provider=provider, model=model)[0]

    # cosine distances
    sims = cosine_similarity([np.array(query_emb)],
                             np.array(art_embs))[0]
    df["distance_cosine"] = 1 - sims
    df = df.sort_values("distance_cosine")

    # ------------------------------------------------------------------
    # Save ranked CSV
    # ------------------------------------------------------------------
    ranked_csv = csv_path.with_name("articles_sorted_by_distance.csv")
    df.to_csv(ranked_csv, index=False)
    print(f"✅ Ranked list saved → {ranked_csv.name}")

    # ------------------------------------------------------------------
    # Bibliometric report
    # ------------------------------------------------------------------
    report_txt = full_report(df, top_n=top_n)
    print("\n" + report_txt)

    rep_path = csv_path.with_name("bibliometric_report.txt")
    with open(rep_path, "w", encoding="utf‑8") as fh:
        fh.write(report_txt)
    print(f"✅ Report saved → {rep_path.name}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted – bye!")
