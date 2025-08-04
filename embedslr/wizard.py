"""
EmbedSLR ‑ interactive command‑line wizard
------------------------------------------

Start with:
    python -m embedslr.wizard

It will:

1. Ask for a Scopus / Web‑of‑Science CSV file.
2. Prompt for the research query.
3. Let you choose an embedding provider + model.
4. Compute cosine distances to the query.
5. Produce three artefacts and bundle them into *embedslr_results.zip*:
      • articles_sorted_by_distance.csv
      • topN_for_metrics.csv         (if you limited Top‑N)
      • biblio_report.txt            (full bibliometric report)
"""

from __future__ import annotations

import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from embedslr.embeddings import get_embeddings, list_models
from embedslr.metrics import full_report


# ─────────────────────────────── helpers ────────────────────────────────
def ask(prompt: str, default: Optional[str] = None) -> str:
    """Input wrapper with optional default value."""
    if default is None:
        return input(f"{prompt}: ").strip()
    reply = input(f"{prompt} [{default}]: ").strip()
    return reply or default


def choose_provider() -> str:
    provs = list(list_models().keys())
    print(f"\nAvailable providers: {', '.join(provs)}")
    prov = ask("Provider", default=provs[0]).lower()
    if prov not in provs:
        sys.exit(f"Unknown provider: {prov}")
    return prov


def choose_model(provider: str) -> str:
    models = list_models()[provider]
    print("\nModels (first 20):")
    for idx, m in enumerate(models[:20], 1):
        print(f"{idx:2d}. {m}")
    return ask(
        "Model (ENTER = 1st or type any model name)",
        default=models[0],
    )


def get_text_columns(frame: pd.DataFrame) -> tuple[str, Optional[str]]:
    """Find suitable Title / Abstract columns in a Scopus/WoS export."""
    title_col = next(
        (c for c in ("Article Title", "Title", "TI") if c in frame.columns), None
    )
    abstr_col = next((c for c in ("Abstract", "AB") if c in frame.columns), None)
    if title_col is None:
        sys.exit("No title column found in the CSV.")
    return title_col, abstr_col


# ───────────────────────────── main wizard ──────────────────────────────
def main() -> None:
    print(">>> EmbedSLR – interactive wizard\n")

    # ------------------------------------------------------------------ CSV
    csv_path = Path(ask("Path to Scopus/WoS CSV file"))
    if not csv_path.is_file():
        sys.exit(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} records, columns: {', '.join(df.columns[:10])} …")

    # ------------------------------------------------------------- basics
    query = ask("Research problem / query")
    provider = choose_provider()
    model = choose_model(provider)

    top_n_str = ask("Top‑N publications for metrics (ENTER = all)")
    top_n = int(top_n_str) if top_n_str else None

    # ----------------------------------------------------------- texts
    title_col, abstr_col = get_text_columns(df)
    texts = (
        df[title_col].fillna("") + " " + df[abstr_col].fillna("")
        if abstr_col
        else df[title_col].fillna("")
    ).tolist()

    # -------------------------------------------------------- embeddings
    print("\nCalculating embedding for the query …")
    q_vec = get_embeddings([query], provider=provider, model=model)[0]

    print("Calculating embeddings for every article …")
    art_vecs = get_embeddings(texts, provider=provider, model=model)

    # ---------------------------------------------------- cosine distance
    sim = cosine_similarity(np.array([q_vec]), np.array(art_vecs))[0]
    df["distance_cosine"] = 1 - sim
    df_sorted = df.sort_values("distance_cosine")

    # -------------------------------------------------------------- files
    out_sorted = csv_path.with_name("articles_sorted_by_distance.csv")
    df_sorted.to_csv(out_sorted, index=False)
    print(f"\nSaved → {out_sorted.name}")

    subset = df_sorted.head(top_n) if top_n else df_sorted
    out_subset = csv_path.with_name("topN_for_metrics.csv")
    subset.to_csv(out_subset, index=False)
    print(f"Saved → {out_subset.name}")

    out_report = csv_path.with_name("biblio_report.txt")
    report_txt = full_report(df_sorted, top_n=top_n, path=out_report)
    print("Saved → biblio_report.txt")

    # -------------------------------------------------------------- zip
    out_zip = csv_path.with_name("embedslr_results.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_sorted, out_sorted.name)
        zf.write(out_subset, out_subset.name)
        zf.write(out_report, out_report.name)
    print(f"Packaged → {out_zip.name}")

    # -------------------------------------------------------------- done
    print("\n✅  Finished. All files are in the current directory.")
    print(textwrap.indent(report_txt, "│ "))


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
