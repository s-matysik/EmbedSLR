"""
EmbedSLR – Colab / Jupyter launcher (widget‑free)
================================================

* Upload CSV (Scopus/WoS) → Query → Provider / Model → liczba TOP N
* Generuje:
  – ranking.csv  (pełna lista)  
  – topN.csv     (N najbardziej podobnych)  
  – biblio_report.txt  (8 wskaźników A–H obliczone na N)  
  wszystko w ZIP → plik pobierany przez `google.colab.files.download()`
"""

from __future__ import annotations
import io, os, sys, zipfile, tempfile, shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from IPython.display import HTML, clear_output, display

IN_COLAB = "google.colab" in sys.modules


# ───────────────────  helpers  ──────────────────────────────────────────────
def _set_api_key(provider: str, key: str):
    env = {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "jina":   "JINA_API_KEY",
        "nomic":  "NOMIC_API_KEY",
    }.get(provider.lower())
    if env:
        os.environ[env] = key


def _all_models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _pipeline(df: pd.DataFrame, query: str, provider: str, model: str,
              save_dir: Path, *, top_n: int | None) -> Path:
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine
    from .bibliometrics import full_report

    # --- przygotowanie tekstów --------------------------------------------
    tcol, acol = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, tcol, acol)

    vecs = get_embeddings(df["combined_text"].tolist(),
                          provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]
    ranked = rank_by_cosine(qvec, vecs, df)

    # --- zapisy ------------------------------------------------------------
    csv_all = save_dir / "ranking.csv"
    ranked.to_csv(csv_all, index=False)

    if top_n is not None:
        csv_top = save_dir / "topN.csv"
        ranked.head(top_n).to_csv(csv_top, index=False)
    else:
        csv_top = None

    rep_path = save_dir / "biblio_report.txt"
    full_report(ranked, path=rep_path, top_n=top_n)

    # --- zip ---------------------------------------------------------------
    zip_path = save_dir / "embedslr_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_all, "ranking.csv")
        if csv_top:
            z.write(csv_top, "topN.csv")
        z.write(rep_path, "biblio_report.txt")
    return zip_path


# ───────────────────  Colab interactive (native upload)  ───────────────────
def _colab_mode(save_dir: Path):
    from google.colab import files  # type: ignore

    display(HTML(
        "<h3>EmbedSLR – interactive upload</h3>"
        "<ol><li><b>Browse</b> → wskaż CSV.</li>"
        "<li>Poczekaj na ✅.</li>"
        "<li>Wpisz odpowiedzi w konsoli.</li></ol>"
    ))

    uploaded = files.upload()
    if not uploaded:
        display(HTML("<b style='color:red'>No file uploaded – abort.</b>"))
        return

    fname, data = next(iter(uploaded.items()))
    df = pd.read_csv(io.BytesIO(data))
    display(HTML(f"✅ Loaded <code>{fname}</code> ({len(df)} rows)<br>"))

    # --- pytania CLI -------------------------------------------------------
    query = input("❓ Research query: ").strip()
    providers = list(_all_models())
    print("Providers:", providers)
    provider = input(f"Provider [default={providers[0]}]: ").strip() or providers[0]

    print("Models for", provider, "→")
    for m in _all_models()[provider]:
        print("  •", m)
    model = input("Model [ENTER=1st]: ").strip() or _all_models()[provider][0]

    top_raw = input("🔢 How many top papers for bibliometrics? [ENTER = all]: ").strip()
    top_n = int(top_raw) if top_raw else None

    key = input("API key (if needed, ENTER=skip): ").strip()
    if key:
        _set_api_key(provider, key)

    print("⏳ Running EmbedSLR …")
    zip_tmp = _pipeline(df, query, provider, model, save_dir, top_n=top_n)

    # przenosimy do /content + download
    dst = Path.cwd() / zip_tmp.name
    shutil.copy(zip_tmp, dst)
    print("✅ Done – file:", dst.name)
    files.download(str(dst))


# ───────────────────  CLI fallback  ────────────────────────────────────────
def _cli_mode(save_dir: Path):
    print("=== EmbedSLR CLI ===")
    csv_p = Path(input("CSV path: ").strip())
    df = pd.read_csv(csv_p)
    query = input("Query: ").strip()
    prov = input("Provider [sbert]: ").strip() or "sbert"
    model = input("Model [ENTER=default]: ").strip() or _all_models()[prov][0]
    top_n = input("Top‑N [ENTER=all]: ").strip()
    top_n = int(top_n) if top_n else None
    key = input("API key [skip]: ").strip()
    if key:
        _set_api_key(prov, key)
    zip_p = _pipeline(df, query, prov, model, save_dir, top_n=top_n)
    print("ZIP saved at", zip_p)


# ───────────────────  public  ──────────────────────────────────────────────
def run(save_dir: str | os.PathLike | None = None):
    save_dir = Path(save_dir or tempfile.mkdtemp(prefix="embedslr_"))
    clear_output()
    if IN_COLAB:
        _colab_mode(save_dir)
    else:
        _cli_mode(save_dir)
