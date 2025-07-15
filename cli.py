"""
embedslr.cli
============

Wiersz‑poleceń dla pakietu *embedslr* — ranking publikacji na podstawie
embeddingów oraz proste metryki bibliometryczne.

Przykłady
---------
# Ranking z lokalnym modelem SBERT
embedslr rank scopus.csv -q "deep learning in medicine"

# Ranking z modelem OpenAI
embedslr rank scopus.csv -q "gene therapy" -e openai -m text-embedding-ada-002

# Wyliczenie metryk bibliometrycznych
embedslr metrics ranking.csv -o report.json
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from embedslr.io.scopus import load_scopus_csv
from embedslr.ranking.ranker import rank_dataframe
from embedslr.biblio.metrics import CorpusMetrics

# -----------------------------------------------------------------------------
# Konfiguracja Typer
# -----------------------------------------------------------------------------
app = typer.Typer(
    add_completion=False,
    help="embedslr – szybki ranking publikacji oraz metryki bibliometryczne "
         "dla systematycznych przeglądów literatury.",
)

# Mapa „alias → klasa” dla silników embeddingowych
_engine_map: dict[str, str] = {
    "local-sbert": "LocalSBERT",
    "openai": "OpenAIEmbedder",
    "cohere": "CohereEmbedder",
    "nomic": "NomicEmbedder",
    "jina": "JinaEmbedder",
}


def _load_engine(alias: str, **kwargs):
    """
    Dynamiczny import i instancjowanie klasy embeddera na podstawie aliasu.

    Parameters
    ----------
    alias : str
        Klucz z `_engine_map` (np. ``"openai"``).
    **kwargs
        Opcjonalne parametry przekazywane do konstruktora embeddera.

    Returns
    -------
    BaseEmbedder
    """
    if alias not in _engine_map:
        raise typer.BadParameter(
            f"Nieznany silnik '{alias}'. Dostępne: {', '.join(_engine_map.keys())}"
        )

    cls_name = _engine_map[alias]
    module = importlib.import_module("embedslr.embeddings")
    cls = getattr(module, cls_name)
    # usuwamy None z kwargs, aby nie przesłaniać parametrów domyślnych
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return cls(**clean_kwargs)


# =============================================================================
# Komenda: rank
# =============================================================================
@app.command()
def rank(
    input_csv: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Plik CSV wyeksportowany ze Scopus/WoS (lub podobny).",
    ),
    query: str = typer.Option(
        ..., "--query", "-q", help="Problem badawczy / zapytanie tekstowe."
    ),
    engine: str = typer.Option(
        "local-sbert",
        "--engine",
        "-e",
        help="Silnik embeddingowy: local-sbert | openai | cohere | nomic | jina",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Nazwa modelu (opcjonalnie)."
    ),
    out: Path = typer.Option(
        "ranking.csv", "--out", "-o", help="Ścieżka pliku wynikowego CSV."
    ),
):
    """
    Tworzy ranking wszystkich rekordów z *input_csv* względem podobieństwa
    kosinusowego do *query*.  Wynik zawiera kolumnę `distance_cosine`
    (0 = identyczne, 2 = przeciwne wektory).
    """
    typer.echo("⏳ Ładowanie danych…")
    df = load_scopus_csv(input_csv)

    typer.echo(f"⚙️ Instancjonowanie embeddera '{engine}' ({model or 'domyślny model'})")
    embedder = _load_engine(engine, model_name=model)

    typer.echo("🔎 Generowanie rankingów…")
    ranked_df = rank_dataframe(df, query=query, embedder=embedder)

    ranked_df.to_csv(out, index=False)
    typer.echo(f"✅ Zapisano {len(ranked_df)} rekordów do: {out.resolve()}")


# =============================================================================
# Komenda: metrics
# =============================================================================
@app.command()
def metrics(
    ranked_csv: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="CSV (posortowany lub surowy) z danymi publikacji.",
    ),
    out: Path = typer.Option(
        "biblio_report.json",
        "--out",
        "-o",
        help="Plik wyjściowy JSON z metrykami bibliometrycznymi.",
    ),
):
    """
    Oblicza podstawowe metryki bibliometryczne spójności zbioru:
    Jaccard słów kluczowych, wspólne referencje, współ‑cytowania itp.
    """
    typer.echo("⏳ Wczytywanie pliku CSV…")
    df = pd.read_csv(ranked_csv, low_memory=False)

    cm = CorpusMetrics(df)

    typer.echo("📊 Obliczanie metryk…")
    report = {
        "avg_jaccard_keywords": cm.jaccard_keywords(),
        # Miejsce na kolejne wskaźniki – można dodać w przyszłości
        # "avg_common_references": cm.common_references(),
        # ...
    }

    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    typer.echo(f"✅ Raport JSON zapisany do: {out.resolve()}")


# =============================================================================
# Punkt wejścia dla setuptools/pyproject
# =============================================================================
def _entry_point():
    """Funkcja uruchamiana przez skrypt konsolowy *embedslr*."""
    try:
        app()
    except Exception as exc:  # pragma: no cover
        typer.echo(f"💥 Błąd: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    _entry_point()
