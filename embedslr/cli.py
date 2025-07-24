"""EmbedSLR – wiersz poleceń."""
import typer
from pathlib import Path
import numpy as np
import pandas as pd

from embeddings.base import get_backend
from ranking.ranker import Ranker

app = typer.Typer(add_completion=False, rich_help_panel="Embed‑SLR CLI")


@app.command()
def embed(
    input_csv: Path = typer.Argument(..., exists=True, readable=True,
                                     help="Plik CSV z kolumnami Title / Abstract"),
    backend: str = typer.Option(
        "local-sbert", "--backend",
        help="local-sbert | openai | cohere | nomic | jina"
    ),
    output: Path = typer.Option(
        "embeddings.npy", "--out", "-o",
        help="Ścieżka do pliku *.npy z macierzą embeddingów"
    ),
):
    """Generuj embeddingi tytuł+abstrakt i zapisz do `output`."""
    df = pd.read_csv(input_csv, low_memory=False)
    texts = (df["Title"].fillna("") + " " + df["Abstract"].fillna("")).tolist()

    model = get_backend(backend)
    vectors = model.encode(texts)
    np.save(output, np.asarray(vectors))

    typer.echo(f"✅  Zapisano {len(vectors)} wektorów  →  {output}")


@app.command()
def rank(
    query: str = typer.Argument(..., help="Zapytanie badawcze"),
    embeddings_file: Path = typer.Argument(
        ..., exists=True, readable=True, help="Plik *.npy z embeddingami"
    ),
    top_k: int = typer.Option(20, "--top", "-k", help="Ile wyników wypisać"),
):
    """Posortuj dokumenty względem zapytania badawczego i wypisz TOP‑k indeksów."""
    embs = np.load(embeddings_file)
    order = Ranker().rank(query, embs)[:top_k]

    typer.echo("🏆  Najbliższe dokumenty (indeksy):")
    for idx in order:
        typer.echo(f"  • {idx}")


# entry‑point zdefiniowany w pyproject.toml
def _entry_point() -> None:  # noqa: D401
    """Uruchom CLI (helper dla setuptools)."""
    app()


if __name__ == "__main__":   # python -m embedslr.cli
    _entry_point()
