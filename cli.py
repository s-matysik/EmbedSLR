import typer, sys, json, yaml, pandas as pd
from pathlib import Path

from embedslr.io.scopus import load_scopus_csv
from embedslr.embeddings.local_sbert import LocalSBERT
from embedslr.ranking.ranker import rank_dataframe
from embedslr.biblio.metrics import CorpusMetrics

app = typer.Typer(add_completion=False,
                  help="embedslr – szybki ranking publikacji + metryki biblio.")

@app.command()
def rank(
    input_csv: Path = typer.Argument(..., exists=True, readable=True),
    query: str      = typer.Option(..., "--query", "-q", help="Problem badawczy"),
    model: str      = typer.Option("all-mpnet-base-v2", help="SBERT model name"),
    out: Path       = typer.Option("ranking.csv", help="Plik wyjściowy CSV")
):
    df = load_scopus_csv(input_csv)
    embedder = LocalSBERT(model_name=model)
    ranked = rank_dataframe(df, query=query, embedder=embedder)
    ranked.to_csv(out, index=False)
    typer.echo(f"Zapisano ranking do {out}")

@app.command()
def metrics(
    ranked_csv: Path = typer.Argument(..., exists=True),
    out: Path        = typer.Option("biblio_report.json")
):
    df = pd.read_csv(ranked_csv)
    metrics = CorpusMetrics(df)
    report = {
        "avg_jaccard_keywords": metrics.jaccard_keywords(),
        # ... dodaj inne wskaźniki …
    }
    json.dump(report, open(out, "w"), indent=2)
    typer.echo(f"Raport zapisany do {out}")

def _entry_point():
    try:
        app()
    except Exception as exc:  # pragma: no cover
        typer.echo(str(exc), err=True)
        sys.exit(1)

if __name__ == "__main__":
    _entry_point()
