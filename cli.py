# fragment embedslr/cli.py  (tylko zmieniona sekcja wyboru silnika)
import typer, importlib

# ...

_engine_map = {
    "local-sbert": "LocalSBERT",
    "openai": "OpenAIEmbedder",
    "cohere": "CohereEmbedder",
    "nomic": "NomicEmbedder",
    "jina": "JinaEmbedder",
}


def _load_engine(tag: str, **kwargs):
    if tag not in _engine_map:
        raise typer.BadParameter(f"Nieznany silnik: {tag}")
    cls_name = _engine_map[tag]
    cls = getattr(importlib.import_module("embedslr.embeddings"), cls_name)
    return cls(**kwargs)


@app.command()
def rank(
    input_csv: Path = typer.Argument(...),
    query: str = typer.Option(..., "--query", "-q"),
    engine: str = typer.Option("local-sbert", help="local-sbert | openai | cohere | nomic | jina"),
    model: str = typer.Option(None, help="Parametr modelu – opcjonalnie"),
    out: Path = typer.Option("ranking.csv"),
):
    # wczytanie ...
    embedder = _load_engine(engine, model_name=model)  # przekazujemy nazwę modelu, jeśli podana
    # reszta bez zmian
