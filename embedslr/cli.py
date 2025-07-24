import typer
from pathlib import Path
from embeddings.base import get_backend
from ranking.ranker import Ranker

app = typer.Typer(add_completion=False, rich_help_panel="Embed‑SLR CLI")

@app.command()
def embed(
    input_csv: Path = typer.Argument(..., exists=True, readable=True,
                                     help="Plik CSV z kolumnami Title / Abstract"),
    backend: str = typer.Option("local-sbert", "--backend",
                                help="Nazwa backendu: local-sbert | openai | cohere | nomic | jina"),
    output: Path = typer.Option("embeddings.npy", "--out", "-o",
                                help="Ścieżka do pliku *.npy z macierzą embeddingów")
):
    """Generuj embeddingi dla tytuł+abstrakt i zapisz do pliku .npy."""
    import pandas as pd, numpy as np
    df = pd.read_csv(input_csv)
    texts = (df["Title"].fillna("") + " " + df["Abstract"].fillna("")).tolist()
    model = get_backend(backend)
    embs = model.encode(texts)
    np.save(output, np.asarray(embs))
    typer.echo(f"✅  Zapisano {len(embs)} wektorów → {output}")

@app.command()
def rank(
    query: str = typer.Argument(..., help="Zapytanie badawcze"),
    embeddings_file: Path = typer.Argument(..., exists=True, readable=True,
                                           help="Plik *.npy z embeddingami (jak z komendy embed)"),
    top_k: int = typer.Option(20, "--top", "-k")
):
    """Posortuj dokumenty względem zapytania badawczego i wypisz TOP‑k."""
    import numpy as np
    embs = np.load(embeddings_file)
    ranker = Ranker()
    order = ranker.rank(query, embs)[:top_k]
    typer.echo("🏆  Najbliższe dokumenty (indeksy):")
    for idx in order:
        typer.echo(f"  • {idx}")

# wewnętrzny punkt wejścia dla setuptools
def _entry_point() -> None:    # noqa: D401
    """Entry‑point wygenerowany w pyproject.toml – nie wywołuj ręcznie."""
    app()

if __name__ == "__main__":     # klik‑to‑run
    _entry_point()
