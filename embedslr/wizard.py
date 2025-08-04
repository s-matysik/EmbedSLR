"""
Interactive command–line “wizard” for EmbedSLR
---------------------------------------------


  • CSV upload (local path)
  • research‑query prompt
  • provider ▸ model selection (all models supported)
  • optional Top‑N filter for the bibliometric report
  • embeddings, cosine‑distance ranking, full report
  • results are written to:
        ├─ articles_sorted_by_distance.csv
        ├─ bibliometric_report.txt
        └─ embedslr_results.zip      (CSV + TXT)

Only English messages are printed.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────
# Internal imports (relative → avoids import‑path issues)
# ────────────────────────────────────────────────────────────
from .embeddings import get_embeddings, list_models           # noqa: E402  (late import style)
from .metrics    import full_report                           # noqa: E402

# ────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────
_POSSIBLE_TITLE   = ["Article Title", "Title", "TI"]
_POSSIBLE_ABSTRACT = ["Abstract", "AB"]


def _find_col(possible: list[str], df: pd.DataFrame) -> str | None:
    for c in possible:
        if c in df.columns:
            return c
    return None


def _combine_title_abs(row, t_col: str | None, a_col: str | None) -> str:
    title = str(row[t_col]) if t_col else ""
    abstr = str(row[a_col]) if a_col else ""
    return f"{title} {abstr}".strip()


def _prompt_number(prompt: str, default: int | None = None) -> int | None:
    txt = input(prompt).strip()
    if not txt:
        return default
    try:
        return int(txt)
    except ValueError:
        return default


# ────────────────────────────────────────────────────────────
# Wizard workflow
# ────────────────────────────────────────────────────────────
def main() -> None:
    print("\nEmbedSLR – command‑line wizard\n")

    # 1. CSV path -------------------------------------------------------------
    csv_path = Path(input("CSV file path: ").strip('"').strip("'")).expanduser()
    if not csv_path.is_file():
        sys.exit(f"ERROR: file not found – {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded file with {len(df)} rows and {len(df.columns)} columns.\n")

    # 2. Research query -------------------------------------------------------
    research_query = input("Research query: ").strip()
    if not research_query:
        sys.exit("ERROR: query cannot be empty.")

    # 3. Provider / model selection ------------------------------------------
    providers = list(list_models().keys())
    print("\nProviders:", ", ".join(providers))
    provider = input(f"Choose provider [default={providers[0]}]: ").strip().lower() or providers[0]
    if provider not in providers:
        sys.exit(f"ERROR: unknown provider '{provider}'.")

    models = list_models()[provider]
    print("\nModels for", provider)
    for i, m in enumerate(models, 1):
        print(f"  {i:>2}. {m}")
    m_idx = _prompt_number(f"Choose model [1]: ", default=1)
    if not (1 <= m_idx <= len(models)):
        sys.exit("ERROR: wrong model number.")
    model_name = models[m_idx - 1]
    print(f"Selected model: {model_name}")

    # 4. Top‑N for report -----------------------------------------------------
    top_n = _prompt_number("\nTop‑N publications for metrics [all]: ", default=None)

    # 5. Prepare combined text -----------------------------------------------
    title_col    = _find_col(_POSSIBLE_TITLE, df)
    abstract_col = _find_col(_POSSIBLE_ABSTRACT, df)
    if not title_col and not abstract_col:
        sys.exit("ERROR: no Title / Abstract column found in CSV.")

    df["combined_text"] = df.apply(_combine_title_abs,
                                   axis=1, t_col=title_col, a_col=abstract_col)

    # 6. Generate embeddings --------------------------------------------------
    print("\n⏳ Embedding query…")
    query_vec = get_embeddings([research_query],
                               provider=provider, model=model_name)[0]

    print("⏳ Embedding documents…")
    doc_vecs = get_embeddings(df["combined_text"].tolist(),
                              provider=provider, model=model_name)

    # 7. Cosine distance ranking ---------------------------------------------
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity([np.array(query_vec)], np.array(doc_vecs))[0]
    df["distance_cosine"] = 1 - sim
    df.sort_values("distance_cosine", inplace=True)

    # 8. Bibliometric report --------------------------------------------------
    report_txt = full_report(df, top_n=top_n)
    print("\n" + report_txt + "\n")

    # 9. Write outputs --------------------------------------------------------
    out_csv  = Path("articles_sorted_by_distance.csv")
    out_txt  = Path("bibliometric_report.txt")

    # JSON‑dump embeddings so the CSV remains portable
    df["combined_embeddings"] = [json.dumps(v) for v in doc_vecs]
    cols = [c for c in df.columns if c not in ("combined_embeddings", "distance_cosine")]
    df = df[cols + ["combined_embeddings", "distance_cosine"]]
    df.to_csv(out_csv, index=False)
    out_txt.write_text(report_txt, encoding="utf‑8")
    print(f"Saved: {out_csv}  •  {out_txt}")

    # 10. Zip bundle ----------------------------------------------------------
    zip_name = "embedslr_results.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv)
        zf.write(out_txt)
    print(f"Packed results → {zip_name}\n")


# ────────────────────────────────────────────────────────────
# Entry‑point
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
