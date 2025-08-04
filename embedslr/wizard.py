"""embedslr.wizard  –  lightweight interactive CLI
===================================================

This “wizard” guides the user through a minimal sequence of steps:

1.  Load a CSV exported e.g. from Scopus / WoS (user‑supplied local path)
2.  Prompt for the *research query* sentence that defines the topic
3.  Let the user pick **provider → model** for text‑embeddings
4.  (Optional) choose *Top‑N* threshold for the bibliometric report
5.  Compute embeddings, cosine‑distance ranking and a full bibliometric report
6.  Save all artefacts to the working directory and pack them into a ZIP

Only English messages are printed to avoid encoding issues on some
terminals.  The heavy lifting is delegated to helper modules of EmbedSLR,
so the file stays short and easy to maintain.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────
# Internal API – reuse, do **not** re‑implement anything here!
# ──────────────────────────────────────────────────────────────
from .embeddings import get_embeddings, list_models
from .io import read_csv, autodetect_columns, combine_title_abstract
from .similarity import rank_by_cosine
from .bibliometrics import full_report

# ──────────────────────────────────────────────────────────────
# Helper utils
# ──────────────────────────────────────────────────────────────
def _prompt(msg: str, default: Optional[str] = None) -> str:
    """Simple input wrapper that handles default answers."""
    tail = f" [default={default}]" if default is not None else ""
    ans = input(f"{msg}{tail}: ").strip()
    return ans or (default or "")


def _pick_from_list(
    name: str, options: list[str], default: Optional[str] = None
) -> str:
    print(f"\n{name} options:")
    for idx, opt in enumerate(options, 1):
        print(f"  {idx:>2}. {opt}")
    ans = _prompt(f"Choose {name.lower()}", default)
    # Allow both index (1‑based) and textual value
    if ans.isdigit():
        i = int(ans) - 1
        if i < 0 or i >= len(options):
            sys.exit(f"ERROR: {name} index out of range.")
        return options[i]
    if ans in options:
        return ans
    sys.exit(f"ERROR: unknown {name.lower()} '{ans}'.")


# ──────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────
def main() -> None:
    print("\nEmbedSLR – command‑line wizard\n")

    # 1. CSV -------------------------------------------------------------------
    csv_path_str = _prompt("CSV file path")
    csv_path = Path(csv_path_str).expanduser()
    if not csv_path.is_file():
        sys.exit(f"ERROR: file not found – {csv_path}")
    df = read_csv(str(csv_path))
    print(f"Loaded file with {len(df)} rows and {len(df.columns)} columns.\n")

    # 2. Detect and combine text columns --------------------------------------
    try:
        title_col, abs_col = autodetect_columns(df)
    except ValueError as e:
        sys.exit(f"{e} – please rename columns or edit the CSV.")
    df["combined_text"] = combine_title_abstract(df, title_col, abs_col)

    # 3. Research query --------------------------------------------------------
    research_query = _prompt("Research query").strip()
    if not research_query:
        sys.exit("ERROR: query cannot be empty.")

    # 4. Provider / model selection -------------------------------------------
    providers = list(list_models().keys())
    provider = _pick_from_list("Provider", providers, default=providers[0])

    models = list_models()[provider]
    model = _pick_from_list("Model", models, default=models[0])

    # 5. Optional Top‑N --------------------------------------------------------
    top_n_raw = _prompt("Top‑N filter for bibliometric report (press Enter for all)")
    top_n: Optional[int] = int(top_n_raw) if top_n_raw.isdigit() else None

    # 6. Embeddings ------------------------------------------------------------
    print("\n⏳ Embedding the research query…")
    query_vec = get_embeddings([research_query], provider=provider, model=model)[0]

    print("⏳ Embedding documents… (this may take a while)")
    doc_vecs = get_embeddings(
        df["combined_text"].tolist(), provider=provider, model=model
    )

    # 7. Rank by cosine distance ----------------------------------------------
    df_ranked = rank_by_cosine(query_vec, doc_vecs, df)

    # 8. Bibliometric report ---------------------------------------------------
    report_txt = full_report(df_ranked, top_n=top_n)
    print("\n" + report_txt + "\n")

    # 9. Persist results -------------------------------------------------------
    out_csv = Path("articles_sorted_by_distance.csv")
    out_txt = Path("bibliometric_report.txt")

    # Store embeddings as JSON strings so the CSV remains self‑contained
    df_ranked["combined_embeddings"] = [json.dumps(v) for v in doc_vecs]
    df_ranked.to_csv(out_csv, index=False)
    out_txt.write_text(report_txt, encoding="utf-8")
    print(f"Saved: {out_csv}  •  {out_txt}")

    # 10. Pack into ZIP --------------------------------------------------------
    zip_name = "embedslr_results.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv)
        zf.write(out_txt)
    print(f"Packed results → {zip_name}\n")


# ──────────────────────────────────────────────────────────────
# Entry‑point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
