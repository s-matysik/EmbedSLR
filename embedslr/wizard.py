"""
Interactive command‑line “wizard” for EmbedSLR.

"""

from __future__ import annotations
import sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from embedslr.embeddings import get_embeddings, list_models
from embedslr.metrics     import full_report


# ───────────────────────────────── helpers ──────────────────────────────────
def _ask(prompt: str, default: str | None = None) -> str:
    txt = input(prompt).strip()
    return txt or (default or "")


def _select_provider() -> tuple[str, list[str]]:
    models_by_provider = list_models()
    providers = list(models_by_provider)
    default   = providers[0]
    p = _ask(f"\n❔  Provider {providers} [default={default}]: ", default).lower()
    while p not in providers:
        p = _ask(f"   → unknown, pick one of {providers}: ", default).lower()
    return p, models_by_provider[p]


def _select_model(provider: str, models: list[str]) -> str:
    print(f"\n📚  Models for {provider} (first 20 shown):")
    for i, m in enumerate(models[:20], 1):
        print(f"  {i:>2}. {m}")
    choice = _ask("Model [ENTER = 1st | index | free text]: ", models[0])
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(models):
            return models[idx - 1]
    return choice


# ───────────────────────────────── wizard ───────────────────────────────────
def main() -> None:
    print("▶  EmbedSLR wizard (local)")
    csv_path = Path(_ask("📄  Path to Scopus/WoS CSV: ")).expanduser()
    if not csv_path.is_file():
        sys.exit(f"! CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"   → loaded {len(df)} rows, columns: {list(df.columns)[:8]} ...")

    query   = _ask("❓  Research problem / query: ")
    provider, avail = _select_provider()
    model    = _select_model(provider, avail)
    top_n_in = _ask("🔢  Top‑N pubs for bibliometrics [ENTER = all]: ")
    top_n    = int(top_n_in) if top_n_in else None

    # ‑‑‑ build combined text (title + abstract) ‑‑‑
    title_col = next((c for c in df.columns if c.lower() in {"title", "article title", "ti"}), None)
    abs_col   = next((c for c in df.columns if c.lower() in {"abstract", "ab"}), None)
    if not title_col:
        sys.exit("! No column with article titles found.")
    df["combined_text"] = (
        df[title_col].fillna("") + " " + (df[abs_col] if abs_col else "")
        .fillna("")
    ).str.strip()

    # ‑‑‑ embeddings ‑‑‑
    print("\n⏳  Computing embedding for the query…")
    q_vec = get_embeddings([query], provider=provider, model=model)[0]

    print("⏳  Computing embeddings for articles…")
    art_vecs = get_embeddings(df["combined_text"].tolist(), provider=provider, model=model)

    # ‑‑‑ cosine distance & sorting ‑‑‑
    sims = cosine_similarity([q_vec], np.asarray(art_vecs))[0]
    df["distance_cosine"] = 1 - sims
    df_sorted = df.sort_values("distance_cosine")

    # files to save
    f_sorted  = Path("articles_sorted_by_distance.csv")
    f_metrics = Path("topN_for_metrics.csv")
    f_report  = Path("biblio_report.txt")
    f_zip     = Path("embedslr_results.zip")

    df_sorted.to_csv(f_sorted, index=False)
    df_sorted.head(top_n or len(df_sorted)).to_csv(f_metrics, index=False)

    # ‑‑‑ full bibliometric report (A … I) ‑‑‑
    report_txt = full_report(pd.read_csv(f_metrics))
    f_report.write_text(report_txt, encoding="utf‑8")

    # ‑‑‑ zip everything for convenience ‑‑‑
    with zipfile.ZipFile(f_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in (f_sorted, f_metrics, f_report):
            zf.write(fn)

    print("\n✅  Finished – files generated:")
    for fn in (f_sorted, f_metrics, f_report, f_zip):
        print(f"   • {fn.resolve()}")

    print("\n📊  ---  BIBLIOMETRIC REPORT  ---")
    print(report_txt)


if __name__ == "__main__":
    main()
