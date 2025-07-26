# embedslr/colab_app.py  –  v0.3.0  (widget‑free)
# MIT License – 2025 EmbedSLR team
# -------------------------------------------------
"""
Uruchomienie:
>>> from embedslr.colab_app import run
>>> run()

Działa w Google Colab oraz w klasycznym Jupyter Lab (CLI fallback).
"""

from __future__ import annotations
import io, os, sys, zipfile, tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
from IPython.display import HTML, clear_output, display

IN_COLAB = "google.colab" in sys.modules


# ---------- małe utilsy ------------------------------------------------------
def _set_api_key(provider: str, key: str) -> None:
    env = {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "jina":   "JINA_API_KEY",
        "nomic":  "NOMIC_API_KEY",
    }.get(provider.lower())
    if env:
        os.environ[env] = key


def _models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _pipeline(df: pd.DataFrame, query: str, provider: str,
              model: str, save_dir: Path) -> Path:
    """Zwraca ścieżkę do ZIP‑a z rankingiem i raportem."""
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine
    from .bibliometrics import full_report

    title, abstr = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, title, abstr)

    vecs = get_embeddings(df["combined_text"].tolist(),
                          provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]
    ranked = rank_by_cosine(qvec, vecs, df)

    csv_path = save_dir / "ranking.csv"
    ranked.to_csv(csv_path, index=False)
    txt_path = save_dir / "biblio_report.txt"
    full_report(ranked, path=txt_path)

    zip_path = save_dir / "embedslr_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="ranking.csv")
        zf.write(txt_path, arcname="biblio_report.txt")
    return zip_path


# ---------- główny tryb Colab -----------------------------------------------
def _colab_mode(save_dir: Path):
    from google.colab import files  # type: ignore

    display(HTML("<h3>EmbedSLR – interactive upload</h3>"
                 "<ol>"
                 "<li><b>Browse</b> → wybierz plik CSV z Scopus/WoS.</li>"
                 "<li>Poczekaj na komunikat ✅.</li>"
                 "<li>Postępuj zgodnie z promptami w konsoli.</li></ol>"))

    uploaded = files.upload()
    if not uploaded:
        display(HTML("<b style='color:red'>Brak pliku – przerywam.</b>"))
        return
    name, data = next(iter(uploaded.items()))
    df = pd.read_csv(io.BytesIO(data))
    display(HTML(f"✅ Wczytano <code>{name}</code> ({len(df)} rekordów)<br>"))

    # pytania w konsoli ↓↓↓
    query = input("❓ Problem badawczy / query: ").strip()
    if not query:
        print("Abort – query nie może być puste.")
        return

    provs = list(_models())
    provider = input(f"Provider {provs} [default={provs[0]}]: ").strip() or provs[0]
    model = input("Model (ENTER = domyślny): ").strip() or _models()[provider][0]
    key = input("API key (jeśli wymagany, ENTER = pomiń): ").strip()
    if key:
        _set_api_key(provider, key)

    print("⏳ Uruchamiam EmbedSLR …")
    zip_path = _pipeline(df, query, provider, model, save_dir)
    display(HTML(f"<h4>✅ Gotowe – pobierz:</h4>"
                 f"<a href='{zip_path}' download>📦 embedslr_results.zip</a>"))


# ---------- fallback CLI (lokalny Jupyter) -----------------------------------
def _cli_mode(save_dir: Path):
    print("== EmbedSLR CLI ==")
    csv_path = input("Ścieżka do CSV: ").strip()
    if not csv_path:
        print("Abort.")
        return
    df = pd.read_csv(csv_path)
    query = input("Query: ").strip()
    prov = input("Provider (sbert/openai/cohere/nomic/jina): ").strip() or "sbert"
    model = input("Model (ENTER=default): ").strip() or _models()[prov][0]
    key = input("API key (ENTER=pomiń): ").strip()
    if key:
        _set_api_key(prov, key)

    zip_path = _pipeline(df, query, prov, model, save_dir)
    print(f"✓ Done. Results in {zip_path}")


# ---------- public -----------------------------------------------------------
def run(save_dir: str | os.PathLike | None = None) -> None:
    """
    Interaktywny upload (Google Colab) lub prosty CLI w Jupyter.
    """
    save_dir = Path(save_dir or tempfile.mkdtemp(prefix="embedslr_"))
    clear_output()
    if IN_COLAB:
        _colab_mode(save_dir)
    else:
        _cli_mode(save_dir)
