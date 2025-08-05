"""
EmbedSLR – Terminal Wizard (local)
==================================
Interactive wizard for running EmbedSLR in a local environment. 
The pipeline (embedding → ranking → full bibliometric report → ZIP).
"""

from __future__ import annotations

import os
import sys
import zipfile
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

@@
 
# ────────── helper functions (extracted from colab_app) ─────────────────
def _env_var(provider: str) -> str | None:
    """Returns the ENV variable name for the API key of the given provider."""
    return {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "jina":   "JINA_API_KEY",
        "nomic":  "NOMIC_API_KEY",
    }.get(provider.lower())


def _ensure_sbert_installed() -> None:
    """
    Ensures the *sentence-transformers* library is available.
    • If missing, prompts the user and installs it (`pip install --user sentence-transformers`).
    • On subsequent runs, installation is skipped if already present.
    """
    try:
        importlib.import_module("sentence_transformers")
    except ModuleNotFoundError:
        ans = _ask(
            "📦  The 'sentence-transformers' package is not installed. Install it now? (y/N)",
            "N",
        ).lower()
        if ans == "y":
            print("⏳  Installing 'sentence-transformers'…")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--user", "--quiet", "sentence-transformers"]
            )
            print("✅  Installation complete.\n")
        else:
            sys.exit("❌  Cannot use the 'sbert' provider without 'sentence-transformers'.")



def _models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _ensure_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures presence of columns:
      • Title
      • Author Keywords
      • Parsed_References  (set[str])
    """
    if "Parsed_References" not in df.columns:
        if "References" in df.columns:
            df["Parsed_References"] = df["References"].fillna("").apply(
                lambda x: {r.strip() for r in x.split(");") if r.strip()}
            )
        else:
            df["Parsed_References"] = [set()] * len(df)

    if "Author Keywords" not in df.columns:
        df["Author Keywords"] = ""

    if "Title" not in df.columns:
        if "Article Title" in df.columns:
            df["Title"] = df["Article Title"]
        else:
            df["Title"] = [f"Paper_{i}" for i in range(len(df))]
    return df


def _pipeline(
    df: pd.DataFrame,
    query: str,
    provider: str,
    model: str,
    out: Path,
    top_n: int | None,
) -> Path:
    """
    Executes the full EmbedSLR workflow and returns the path to the ZIP of results.
    """
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine
    from .bibliometrics import full_report

    df = _ensure_aux_columns(df.copy())

    # 1. Prepare text for embedding
    tcol, acol = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, tcol, acol)

    # 2. Embeddings
    vecs = get_embeddings(df["combined_text"].tolist(),
                          provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]

    # 3. Ranking
    ranked = rank_by_cosine(qvec, vecs, df)

    # 4. Save ranking.csv
    out.mkdir(parents=True, exist_ok=True)
    p_all = out / "ranking.csv"
    ranked.to_csv(p_all, index=False)

    # 5. Top‑N (optional)
    p_top = None
    if top_n:
        p_top = out / "topN.csv"
        ranked.head(top_n).to_csv(p_top, index=False)

    # 6. Full bibliometric report
    rep = out / "biblio_report.txt"
    full_report(ranked, path=rep, top_n=top_n)

    # 7. ZIP with results
    zf = out / "embedslr_results.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p_all, "ranking.csv")
        if p_top:
            z.write(p_top, "topN.csv")
        z.write(rep, "biblio_report.txt")
    return zf


# ────────── simple CLI ────────────────────────────────────────────────────
def _ask(prompt: str, default: Optional[str] = None) -> str:
    msg = f"{prompt}"
    if default is not None:
        msg += f" [{default}]"
    msg += ": "
    ans = input(msg).strip()
    return ans or (default or "")


def _select_provider() -> str:
    provs = list(_models())
    print("📜  Available providers:", ", ".join(provs))
    return _ask("Provider", provs[0])

def _select_model(provider: str) -> str:
    mods = _models()[provider]
    print(f"📜  Models for {provider} (first 20):")
    for m in mods[:20]:
        print("   •", m)
    return _ask("Model", mods[0])


def run(save_dir: str | os.PathLike | None = None):
    """
    Runs the EmbedSLR wizard in terminal/screen/tmux.
    """
    print("\n== EmbedSLR Wizard (local) ==\n")

    # Input file
    csv_path = Path(_ask("📄  Path to CSV file")).expanduser()
    if not csv_path.exists():
        sys.exit(f"❌  File not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅  Loaded {len(df)} records\n")

    # Analysis parameters
    query = _ask("❓  Research query").strip()
    provider = _select_provider()
    model = _select_model(provider)

  # Ensure local provider prerequisites
    if provider.lower() == "sbert":
        _ensure_sbert_installed()

    model = _select_model(provider)

    # Optional: one‑time download / verification of the local model
    if provider.lower() == "sbert":
        from sentence_transformers import SentenceTransformer
        print(f"⏳  Checking/downloading model '{model}' to local cache…")
        SentenceTransformer(model)  # after this, offline use is possible

    
    n_raw = _ask("🔢  Top‑N publications for metrics (ENTER = all)")
    top_n = int(n_raw) if n_raw else None

    # API key (if needed)
    key_env = _env_var(provider)
    if key_env and not os.getenv(key_env):
        key = _ask(f"🔑  {key_env} (ENTER = skip)")
        if key:
            os.environ[key_env] = key

    # Output folder
    out_dir = Path(save_dir or os.getcwd()).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    print("\n⏳  Processing…")
    zip_path = _pipeline(
        df=df,
        query=query,
        provider=provider,
        model=model,
        out=out_dir,
        top_n=top_n,
    )

    print("\n✅  Done!")
    print("📁  Results saved to:", out_dir)
    print("🎁  ZIP package:", zip_path)
    print("   (ranking.csv, topN.csv – if selected, biblio_report.txt)\n")


if __name__ == "__main__":
    run()
