#!/usr/bin/env python3
"""
EmbedSLR – Terminal Wizard (local)
==================================

Interaktywny kreator do uruchamiania EmbedSLR w środowisku lokalnym
(terminal, screen, tmux itp.).  Pipeline (embedding → ranking →
pełny raport bibliometryczny → ZIP) odtwarza dokładnie te kroki,
które wykonuje colab_app.py, lecz bez zależności od IPython/Colab.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ────────── pomocnicze funkcje (wyjęte z colab_app) ──────────────────────
def _env_var(provider: str) -> str | None:
    """Zwraca nazwę zmiennej ENV dla klucza API danego providera."""
    return {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "jina":   "JINA_API_KEY",
        "nomic":  "NOMIC_API_KEY",
    }.get(provider.lower())


def _models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _ensure_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gwarantuje obecność kolumn:
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
    Realizuje pełny workflow EmbedSLR i zwraca ścieżkę do ZIP‑a z wynikami.
    """
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine
    from .bibliometrics import full_report

    df = _ensure_aux_columns(df.copy())

    # 1. Tekst wejściowy dla embeddingu
    tcol, acol = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, tcol, acol)

    # 2. Embeddingi
    vecs = get_embeddings(df["combined_text"].tolist(),
                          provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]

    # 3. Ranking
    ranked = rank_by_cosine(qvec, vecs, df)

    # 4. Zapis ranking.csv
    out.mkdir(parents=True, exist_ok=True)
    p_all = out / "ranking.csv"
    ranked.to_csv(p_all, index=False)

    # 5. Top‑N (opcjonalnie)
    p_top = None
    if top_n:
        p_top = out / "topN.csv"
        ranked.head(top_n).to_csv(p_top, index=False)

    # 6. Pełny raport bibliometryczny
    rep = out / "biblio_report.txt"
    full_report(ranked, path=rep, top_n=top_n)

    # 7. ZIP z wynikami
    zf = out / "embedslr_results.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p_all, "ranking.csv")
        if p_top:
            z.write(p_top, "topN.csv")
        z.write(rep, "biblio_report.txt")
    return zf


# ────────── proste CLI ────────────────────────────────────────────────────
def _ask(prompt: str, default: Optional[str] = None) -> str:
    msg = f"{prompt}"
    if default is not None:
        msg += f" [{default}]"
    msg += ": "
    ans = input(msg).strip()
    return ans or (default or "")


def _select_provider() -> str:
    provs = list(_models())
    print("📜  Dostępni providerzy:", ", ".join(provs))
    return _ask("Provider", provs[0])


def _select_model(provider: str) -> str:
    mods = _models()[provider]
    print(f"📜  Modele dla {provider} (pierwsze 20):")
    for m in mods[:20]:
        print("   •", m)
    return _ask("Model", mods[0])


def run(save_dir: str | os.PathLike | None = None):
    """
    Uruchamia kreator EmbedSLR w terminalu / screen / tmux.
    """
    print("\n== EmbedSLR Wizard (local) ==\n")

    # Plik wejściowy
    csv_path = Path(_ask("📄  Ścieżka do pliku CSV")).expanduser()
    if not csv_path.exists():
        sys.exit(f"❌  Nie znaleziono pliku: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅  Załadowano {len(df)} rekordów\n")

    # Parametry analizy
    query = _ask("❓  Research query").strip()
    provider = _select_provider()
    model = _select_model(provider)
    n_raw = _ask("🔢  Top‑N publikacji do metryk (ENTER = wszystkie)")
    top_n = int(n_raw) if n_raw else None

    # Klucz API (jeśli potrzebny)
    key_env = _env_var(provider)
    if key_env and not os.getenv(key_env):
        key = _ask(f"🔑  {key_env} (ENTER = pomiń)")
        if key:
            os.environ[key_env] = key

    # Folder wyjściowy
    out_dir = Path(save_dir or os.getcwd()).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Uruchomienie pipeline
    print("\n⏳  Przetwarzanie …")
    zip_path = _pipeline(
        df=df,
        query=query,
        provider=provider,
        model=model,
        out=out_dir,
        top_n=top_n,
    )

    print("\n✅  Gotowe!")
    print("📁  Wyniki zapisane w :", out_dir)
    print("🎁  Paczka ZIP        :", zip_path)
    print("   (ranking.csv, topN.csv – jeśli wybrano, biblio_report.txt)\n")


if __name__ == "__main__":
    run()
