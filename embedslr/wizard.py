"""
Interactive CLI “wizard” for EmbedSLR
-------------------------------------

This script is intentionally *thin*: it only orchestrates I/O and user
interaction.  All heavy lifting is delegated to the public APIs exposed in

* embedslr.embeddings      – get_embeddings, list_models
* embedslr.metrics / bibliometric – full_report

so that **one single implementation of every feature** is shared between the
Colab notebook and the local CLI.

Run with::

    python -m embedslr.wizard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import indent

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports that *must* already exist in the package
# ---------------------------------------------------------------------------
from embedslr.embeddings import get_embeddings, list_models

# full_report lives either in metrics.py (newer versions) or bibliometric.py
try:
    from embedslr.metrics import full_report  # type: ignore
except ModuleNotFoundError:                   # pragma: no cover
    from embedslr.bibliometric import full_report  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ask(question: str, default: str | None = None) -> str:
    """Prompt the user; return reply or *default* if blank."""
    prompt = f"{question.strip()} "
    if default is not None:
        prompt += f"[ENTER = {default}] "
    reply = input(prompt).strip()
    return reply if reply else (default or "")


def _choose_from_list(title: str, options: list[str], default_idx: int = 0) -> str:
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i:>2}) {opt}")
    while True:
        sel = _ask("Choice number", str(default_idx + 1))
        if sel.isdigit() and 1 <= int(sel) <= len(options):
            return options[int(sel) - 1]
        print("Invalid selection – please try again.")


def _combine_title_abstract(df: pd.DataFrame) -> pd.Series:
    possible_title = ["Article Title", "Title", "TI"]
    possible_abstr = ["Abstract", "AB"]

    def first(colnames):
        return next((c for c in colnames if c in df.columns), None)

    t_col = first(possible_title)
    a_col = first(possible_abstr)
    if not t_col and not a_col:
        sys.exit("❌  CSV file lacks both title *and* abstract columns.")

    def _row(r):
        return f"{r.get(t_col,'')} {r.get(a_col,'')}".strip()

    return df.apply(_row, axis=1)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901  (complexity is fine for a tiny CLI)
    print("EmbedSLR – interactive CLI\n")

    # 1. Load CSV ------------------------------------------------------------
    csv_path = Path(_ask("Path to Scopus / Web‑of‑Science CSV file")).expanduser()
    if not csv_path.is_file():
        sys.exit(f"❌  File not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅  Loaded {len(df)} records from «{csv_path.name}»")

    # 2. Research question ---------------------------------------------------
    query = _ask("Research question / problem statement")
    if not query:
        sys.exit("❌  Research question must not be empty.")

    # 3. Provider & model ----------------------------------------------------
    models_by_provider = list_models()
    provider = _choose_from_list("Select embedding provider:",
                                 sorted(models_by_provider))
    model = _choose_from_list(f"Models for {provider}:",
                              models_by_provider[provider])

    top_n_str = _ask("Top‑N lowest‑distance papers to include in metrics "
                     "(ENTER = all)", "")
    top_n = int(top_n_str) if top_n_str.isdigit() else None

    # 4. Build documents -----------------------------------------------------
    df["combined_text"] = _combine_title_abstract(df)
    texts = df["combined_text"].tolist()

    # 5. Embeddings ----------------------------------------------------------
    print("\n⏳  Generating embeddings…")
    article_embs = get_embeddings(texts, provider=provider, model=model)
    query_emb    = get_embeddings([query], provider=provider, model=model)[0]

    # 6. Cosine distances ----------------------------------------------------
    sims = (np.dot(article_embs, query_emb) /
            (np.linalg.norm(article_embs, axis=1) * np.linalg.norm(query_emb)))
    df["distance_cosine"] = 1 - sims

    # 7. Sort + save ---------------------------------------------------------
    df_sorted = df.sort_values("distance_cosine")
    out_csv = csv_path.with_suffix(".sorted.csv")
    df_sorted.to_csv(out_csv, index=False)
    print(f"📄  Saved sorted results → {out_csv}")

    # 8. Bibliometric report -------------------------------------------------
    report = full_report(df_sorted, top_n=top_n)
    out_txt = csv_path.with_suffix(".report.txt")
    Path(out_txt).write_text(report, encoding="utf‑8")
    print(f"\n{indent(report, '│ ')}")
    print(f"\n📄  Saved report        → {out_txt}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
