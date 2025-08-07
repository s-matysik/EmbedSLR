from __future__ import annotations
import argparse, json, sys, os
import pandas as pd

from .io import read_csv, autodetect_columns, combine_title_abstract
from .embeddings import get_embeddings
from .similarity import rank_by_cosine
from .bibliometrics import full_report


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="embedslr", description="SLR screening toolkit")
    ap.add_argument("-i", "--input", required=True, help="CSV file exported from Scopus")
    ap.add_argument("-q", "--query", required=True, help="Research problem / query string")
    ap.add_argument("-p", "--provider", default="sbert",
                    choices=["sbert", "openai", "cohere", "nomic", "jina"])
    ap.add_argument("-m", "--model", help="Override default model name")
    ap.add_argument("--api_key", help="Pass API key via CLI (otherwise use env var)")
    ap.add_argument("-o", "--out", default="ranking.csv")
    ap.add_argument("-r", "--report", default="biblio_report.txt")
    ap.add_argument("--json-embs", action="store_true", help="Save embeddings column")
    return ap


def main(argv: list[str] | None = None):
    args = _parser().parse_args(argv)

    if args.api_key:
        var = {
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "nomic": "NOMIC_API_KEY",
            "jina": "JINA_API_KEY",
        }[args.provider]
        os.environ[var] = args.api_key

    df = read_csv(args.input)
    title_col, abs_col = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, title_col, abs_col)

    print("[+] Generating embeddings …")
    embs = get_embeddings(df["combined_text"].tolist(),
                          provider=args.provider, model=args.model)
    qvec = get_embeddings([args.query], provider=args.provider, model=args.model)[0]

    ranked = rank_by_cosine(qvec, embs, df)
    if args.json_embs:
        ranked["combined_embeddings"] = [json.dumps(e) for e in embs]

    ranked.to_csv(args.out, index=False)
    print(f"[✓] Ranking saved -> {args.out}")

    if args.report:
        txt = full_report(ranked, path=args.report)
        print(txt)
        print(f"[✓] Bibliometric report -> {args.report}")


if __name__ == "__main__":
    main()
