"""embedslr.wizard – interactive CLI for EmbedSLR
=================================================

Loads a bibliographic CSV, embeds records, ranks them by cosine distance to
a user query and produces a bibliometric report + sorted CSV + ZIP package.

Key fixes compared with the previous revision:
* Automatic extraction of reference / keyword lists (df['refs'], df['kws'])
* The bibliometric report is run on Top‑N ranked records, not the full set
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .embeddings import get_embeddings, list_models
from .io import read_csv, autodetect_columns, combine_title_abstract
from .similarity import rank_by_cosine
from .bibliometrics import full_report

# ──────────────────────────────────────────────────────────────
# Helpers – user interaction
# ──────────────────────────────────────────────────────────────
def _prompt(msg: str, default: Optional[str] = None) -> str:
    tail = f" [default={default}]" if default else ""
    ans = input(f"{msg}{tail}: ").strip()
    return ans or (default or "")


def _pick_from_list(name: str, options: list[str], default: Optional[str]) -> str:
    print(f"\n{name} options:")
    for idx, opt in enumerate(options, 1):
        print(f"  {idx:>2}. {opt}")
    choice = _prompt(f"Choose {name.lower()}", default)
    if choice.isdigit():
        idx = int(choice) - 1
        if not 0 <= idx < len(options):
            sys.exit(f"ERROR: {name} index out of range.")
        return options[idx]
    if choice in options:
        return choice
    sys.exit(f"ERROR: unknown {name.lower()} '{choice}'.")


# ──────────────────────────────────────────────────────────────
# Helpers – data preparation
# ──────────────────────────────────────────────────────────────
_REF_COLS = {"references", "refs", "reference list"}
_KW_COLS = {
    "author keywords",
    "index keywords",
    "keywords",
    "authkeywords",
    "keyword list",
}


def _extract_lists_from_string(s: str | float) -> List[str]:
    """Split a Scopus / WoS reference or keyword field into a list."""
    if not isinstance(s, str):
        return []
    # Scopus separates with '; '  whereas WoS often uses ';' or ','
    parts = re.split(r";\s*|,\s*", s)
    return [p.strip() for p in parts if p.strip()]


def _add_refs_and_kws_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Find reference / keyword columns and create df['refs'] & df['kws']."""
    lower_cols = {c.lower(): c for c in df.columns}
    # References
    ref_col = next((lower_cols[c] for c in _REF_COLS if c in lower_cols), None)
    if ref_col:
        df["refs"] = df[ref_col].apply(_extract_lists_from_string)
    else:
        df["refs"] = [[]] * len(df)

    # Keywords – merge author & index if both exist
    kw_cols_found = [lower_cols[c] for c in _KW_COLS if c in lower_cols]
    if kw_cols_found:
        df["kws"] = (
            df[kw_cols_found]
            .astype(str)
            .agg("; ".join, axis=1)
            .apply(_extract_lists_from_string)
        )
    else:
        df["kws"] = [[]] * len(df)

    # Warn if nearly all lists are empty – indicates wrong column mapping
    empty_refs = df["refs"].str.len().le(0).sum()
    empty_kws = df["kws"].str.len().le(0).sum()
    if empty_refs > 0.95 * len(df):
        print(
            "⚠️  Warning: >95 % rows have no references. "
            "Check if the reference column name is recognised."
        )
    if empty_kws > 0.95 * len(df):
        print(
            "⚠️  Warning: >95 % rows have no keywords. "
            "Check if the keyword column name is recognised."
        )
    return df


# ──────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────
def main() -> None:
    print("\nEmbedSLR – command‑line wizard\n")

    # 1. Read CSV --------------------------------------------------------------
    csv_path = Path(_prompt("CSV file path")).expanduser()
    if not csv_path.is_file():
        sys.exit(f"ERROR: file not found – {csv_path}")
    df = read_csv(str(csv_path))
    print(f"Loaded file with {len(df)} rows and {len(df.columns)} columns.\n")

    # 2. Detect title & abstract; create combined_text -------------------------
    try:
        title_col, abs_col = autodetect_columns(df)
    except ValueError as e:
        sys.exit(f"{e} – please rename columns or edit the CSV.")
    df["combined_text"] = combine_title_abstract(df, title_col, abs_col)

    # 3. Add refs + kws columns (critical for full_report) ---------------------
    df = _add_refs_and_kws_columns(df)

    # 4. Research query --------------------------------------------------------
    query = _prompt("Research query").strip()
    if not query:
        sys.exit("ERROR: query cannot be empty.")

    # 5. Provider / model ------------------------------------------------------
    providers = list(list_models().keys())
    provider = _pick_from_list("Provider", providers, default=providers[0])
    models = list_models()[provider]
    model = _pick_from_list("Model", models, default=models[0])

    # 6. Top‑N filter ----------------------------------------------------------
    top_n_str = _prompt("Top‑N filter for bibliometric report (press Enter for all)")
    top_n = int(top_n_str) if top_n_str.isdigit() else None

    # 7. Embeddings ------------------------------------------------------------
    print("\n⏳ Embedding the research query…")
    query_vec = get_embeddings([query], provider=provider, model=model)[0]

    print("⏳ Embedding documents… (this may take a while)")
    doc_vecs = get_embeddings(df["combined_text"].tolist(), provider=provider, model=model)

    # 8. Rank by cosine distance ----------------------------------------------
    df_ranked = rank_by_cosine(query_vec, doc_vecs, df)

    # 9. Slice Top‑N for the report -------------------------------------------
    if top_n is not None and 0 < top_n < len(df_ranked):
        df_report = df_ranked.head(top_n).copy()
    else:
        df_report = df_ranked.copy()

    # 10. Bibliometric report --------------------------------------------------
    report_txt = full_report(df_report)
    print("\n==== BIBLIOMETRIC REPORT (Top‑{}) ====\n".format(top_n or "ALL"))
    print(report_txt + "\n")

    # 11. Persist results ------------------------------------------------------
    out_csv = Path("articles_sorted_by_distance.csv")
    out_txt = Path("bibliometric_report.txt")
    df_ranked["combined_embeddings"] = [json.dumps(v) for v in doc_vecs]
    df_ranked.to_csv(out_csv, index=False)
    out_txt.write_text(report_txt, encoding="utf-8")
    print(f"Saved: {out_csv}  •  {out_txt}")

    # 12. Pack everything into ZIP --------------------------------------------
    zip_name = "embedslr_results.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv)
        zf.write(out_txt)
    print(f"Packed results → {zip_name}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
