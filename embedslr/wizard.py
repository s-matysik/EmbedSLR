"""
Interaktywny kreator EmbedSLR
Uruchom:  $ embedslr-wizard
"""
from __future__ import annotations

import os
import sys
import zipfile
import shutil
import tempfile
import textwrap
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import list_models, get_embeddings

# ╭──────────────────────────────────────────────────────╮
# │  1.  Pobranie danych wejściowych                     │
# ╰──────────────────────────────────────────────────────╯
print("📦  Ścieżka do pliku CSV ze Scopus/WoS:")
csv_path = Path(input(">> ").strip()).expanduser()
if not csv_path.exists():
    sys.exit(f"❌  Nie znaleziono pliku: {csv_path}")

df = pd.read_csv(csv_path, low_memory=False)
print(f"✅  Załadowano {len(df)} rekordów, kolumny: {list(df.columns)[:8]}...")

query = input("❓  Podaj problem badawczy / query:\n>> ").strip()

# ╭──────────────────────────────────────────────────────╮
# │  2.  Wybór providera i modelu                       │
# ╰──────────────────────────────────────────────────────╯
prov_list = list(list_models().keys())
print("\n📜  Dostępni providerzy:", prov_list)
provider = input(f"Provider [default={prov_list[0]}]: ").strip() or prov_list[0]

models = list_models()[provider]
print(f"\n📜  Modele dla {provider}  (pierwsze 20):")
for i, m in enumerate(models[:20], 1):
    print(f"  {i:2d}. {m}")
model = input("Model [ENTER = 1‑szy z listy lub dowolna nazwa]: ").strip() or models[0]

# ╭──────────────────────────────────────────────────────╮
# │  3.  Top‑N + klucze API                             │
# ╰──────────────────────────────────────────────────────╯
try:
    topN = int(input("🔢  Top‑N publikacji do analizy bibliometrycznej [ENTER = wszystkie]: ") or 0)
except ValueError:
    topN = 0

need_key = provider in {"openai", "cohere", "nomic", "jina"}
if need_key and not os.getenv(f"{provider.upper()}_API_KEY"):
    key = input(f"🔑  Podaj {provider.upper()}_API_KEY (ENTER = pomiń): ").strip()
    if key:
        os.environ[f"{provider.upper()}_API_KEY"] = key

# ╭──────────────────────────────────────────────────────╮
# │  4.  Przygotowanie tekstów                          │
# ╰──────────────────────────────────────────────────────╯
title_col = next((c for c in ("Article Title", "Title", "TI") if c in df.columns), None)
abstr_col = next((c for c in ("Abstract", "AB") if c in df.columns), None)
if not title_col:
    sys.exit("❌  Nie znaleziono kolumny z tytułem (Title).")

df["combined_text"] = (
    df[title_col].fillna("").astype(str) + " " +
    (df[abstr_col].fillna("").astype(str) if abstr_col else "")
)
texts = df["combined_text"].tolist()

# ╭──────────────────────────────────────────────────────╮
# │  5.  Embeddingi i distance_cosine                   │
# ╰──────────────────────────────────────────────────────╯
print("\n⏳  Liczę embedding dla zapytania…")
emb_q = np.array(get_embeddings([query], provider=provider, model=model)[0])

print("⏳  Liczę embeddingi dla artykułów…")
emb_a = np.array(get_embeddings(texts, provider=provider, model=model))

dist = 1 - cosine_similarity([emb_q], emb_a)[0]
df["distance_cosine"] = dist
df_sorted = df.sort_values("distance_cosine")

# ╭──────────────────────────────────────────────────────╮
# │  6.  Top‑N i zapis CSV                              │
# ╰──────────────────────────────────────────────────────╯
out_dir = Path.cwd()
sorted_csv = out_dir / "articles_sorted_by_distance.csv"
df_sorted.to_csv(sorted_csv, index=False)
print(f"📄  Zapisano {sorted_csv}")

if topN and topN < len(df_sorted):
    df_top = df_sorted.head(topN)
else:
    df_top = df_sorted

top_csv = out_dir / "topN_for_metrics.csv"
df_top.to_csv(top_csv, index=False)
print(f"📄  Zapisano {top_csv}")

# ── standaryzacja kolumn ────────────────────────────────────────────────
# tytuł
if "Title" not in df_top.columns and "Article Title" in df_top.columns:
    df_top = df_top.copy()
    df_top.rename(columns={"Article Title": "Title"}, inplace=True)

# słowa kluczowe
if "Author Keywords" not in df_top.columns:
    df_top["Author Keywords"] = ""

# referencje → Parsed_References jako sety
if "Parsed_References" not in df_top.columns:
    import re

    ref_col = next((c for c in ("References", "Cited References") if c in df_top.columns), None)
    if ref_col:
        def _parse(refs: str | float) -> set[str]:
            if not isinstance(refs, str):
                return set()
            return {p.lower().strip() for p in re.split(r";|\n", refs) if p.strip()}

        df_top["Parsed_References"] = df_top[ref_col].apply(_parse)
    else:
        df_top["Parsed_References"] = [set()] * len(df_top)

# ╭──────────────────────────────────────────────────────╮
# │  7.  Bibliometrics (10 wskaźników)                  │
# ╰──────────────────────────────────────────────────────╯
from embedslr.bibliometrics import full_report

report_path = out_dir / "biblio_report.txt"
report_txt = full_report(df_top, path=report_path)  # zapis + zwrot stringa
print(report_txt)
print(f"📄  Zapisano {report_path}")

# ╭──────────────────────────────────────────────────────╮
# │  8.  ZIP‑pakiet do pobrania                         │
# ╰──────────────────────────────────────────────────────╯
zip_path = out_dir / "embedslr_results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in (sorted_csv, top_csv, report_path):
        z.write(p, arcname=p.name)
print(f"🎁  Gotowe – {zip_path}")

print("\n✔️  KONIEC.  Pliki znajdziesz w bieżącym katalogu.")
