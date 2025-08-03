"""
embedslr.wizard
===============

Interaktywny kreator do uruchamiania EmbedSLR poza Google Colab.
• wczytuje eksport Scopus/WoS (CSV)
• pobiera zapytanie badawcze
• pozwala wybrać dostawcę i model embeddingów
• opcjonalnie zawęża Top‑N publikacji do analizy bibliometrycznej
• tworzy komplet wyników (CSV + raport + ZIP)

Uruchom:
    python -m embedslr.wizard
"""

from __future__ import annotations

import json
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import PROVIDERS, get_embeddings, list_models
from .metrics import full_report


# ───────────────────────────────────────
# Pomocnicze funkcje CLI
# ───────────────────────────────────────
def _choose(prompt: str, options: List[str], default: str | None = None) -> str:
    """Prosty wybór z listy opcji."""
    while True:
        print(prompt)
        for i, o in enumerate(options, 1):
            print(f"  {i}. {o}")
        ans = input(f"⇒ [ENTER = {default or '1'}] ").strip()
        if not ans:
            return default or options[0]
        # numer
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1]
        # nazwa
        if ans in options:
            return ans
        print("✖ Niepoprawny wybór – spróbuj ponownie.\n")


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _combine_title_abstract(df: pd.DataFrame) -> pd.Series:
    """Zwraca serię 'tytuł + abstrakt'."""
    title_col = _detect_column(df, ["Article Title", "Title", "TI"])
    abs_col = _detect_column(df, ["Abstract", "AB"])
    if title_col is None:
        raise RuntimeError("❌ Nie wykryto kolumny z tytułem w CSV.")
    print(f"✓ Używam kolumny '{title_col}'" +
          (f" oraz '{abs_col}'" if abs_col else "") + ".")
    out = df[title_col].astype(str)
    if abs_col:
        out = out + " " + df[abs_col].astype(str)
    return out


# ───────────────────────────────────────
# Minimalny parser referencji (jak w Colab)
# ───────────────────────────────────────
def _parse_references_if_missing(df: pd.DataFrame) -> None:
    if "Parsed_References" in df.columns or "References" not in df.columns:
        return

    def _old_parse(r: str) -> set[str]:
        if pd.isna(r):
            return set()
        parts = [s.strip() for s in str(r).split(");") if s.strip()]
        return set(parts)

    df["Parsed_References"] = df["References"].apply(_old_parse)


# ───────────────────────────────────────
# Główna logika kreatora
# ───────────────────────────────────────
def main() -> None:  # noqa: C901
    print("\n📄  Ścieżka do pliku CSV z Scopus/WoS:")
    csv_path = Path(input("> ").strip())
    if not csv_path.exists():
        sys.exit("Plik nie istnieje – przerwano.")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✓ Załadowano {len(df):,} rekordów.")

    query = input("❓  Podaj problem badawczy / query: ").strip()
    if not query:
        sys.exit("Brak zapytania – przerwano.")

    provider = _choose("\n🌐  Dostępni providerzy:", sorted(PROVIDERS))
    model_list = list_models()[provider]
    show = model_list[:20]  # nie spamujemy całej listy
    model_prompt = ("\n🧠  Modele dla "
                    f"{provider} (pierwsze 20):\n" +
                    "\n".join(f"  {i+1}. {m}" for i, m in enumerate(show)) +
                    "\nModel [ENTER = 1‑szy z listy lub dowolna nazwa]: ")
    chosen = input(model_prompt).strip()
    model = (show[0] if not chosen or chosen.isdigit() and int(chosen) == 1
             else (show[int(chosen)-1] if chosen.isdigit() else chosen))

    topn_str = input("🔢  Top‑N publikacji do analizy bibliometrycznej "
                     "[ENTER = wszystkie]: ").strip()
    top_n = int(topn_str) if topn_str else None

    # ------------------------------------------------------------------ EMB
    texts = _combine_title_abstract(df).tolist()

    print("\n⏳  Liczę embedding dla zapytania…")
    query_emb = get_embeddings([query], provider=provider, model=model)[0]

    print("⏳  Liczę embeddingi dla artykułów…")
    art_embs = get_embeddings(texts, provider=provider, model=model)
    if len(art_embs) != len(texts):
        sys.exit("❌ Embedding count mismatch – przerwano.")

    # ------------------------------------------------------------------ COSINE
    sims = cosine_similarity([np.array(query_emb)], np.array(art_embs))[0]
    df["distance_cosine"] = 1 - sims
    df["combined_embeddings"] = [json.dumps(v) for v in art_embs]
    df_sorted = df.sort_values("distance_cosine")

    # zapisy
    out_sorted = "articles_sorted_by_distance.csv"
    df_sorted.to_csv(out_sorted, index=False)
    print(f"✓ Zapisano {out_sorted}")

    df_for_metrics = df_sorted.head(top_n) if top_n else df_sorted
    _parse_references_if_missing(df_for_metrics)

    out_topn = "topN_for_metrics.csv"
    df_for_metrics.to_csv(out_topn, index=False)
    print(f"✓ Zapisano {out_topn}")

    # ------------------------------------------------------------------ REPORT
    report_txt = full_report(df_for_metrics, path="biblio_report.txt")
    print(report_txt)
    print("✓ Zapisano biblio_report.txt")

    # ------------------------------------------------------------------ ZIP
    with zipfile.ZipFile("embedslr_results.zip", "w",
                         compression=zipfile.ZIP_DEFLATED) as zf:
        for f in (out_sorted, out_topn, "biblio_report.txt"):
            zf.write(f)
    print("📦 Gotowe – embedslr_results.zip")

    print("\n✅ KONIEC. Pliki znajdziesz w bieżącym katalogu.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nPrzerwano przez użytkownika.")
