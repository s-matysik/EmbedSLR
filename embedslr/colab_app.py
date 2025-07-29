"""
EmbedSLR – Google Colab / Jupyter launcher (widget‑free)
v0.5.0 – 2025‑07‑26
--------------------------------------------------------
• Upload CSV → Query → Provider + Model (lista pełna) → Top N
• Wyniki:
    ranking.csv    – pełna lista
    topN.csv       – N publikacji o najmniejszym distance_cosine (opcjonalnie)
    biblio_report.txt  – 8 wskaźników A–H
    ➜ wszystko w ZIP, który pobierze się automatycznie
"""
from __future__ import annotations
import io, os, sys, tempfile, zipfile, shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from IPython.display import HTML, clear_output, display

IN_COLAB = "google.colab" in sys.modules


# ───────────── helpers ──────────────────────────────────────────────────────
def _env_var(provider: str) -> str | None:
    return {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "jina":   "JINA_API_KEY",
        "nomic":  "NOMIC_API_KEY",
    }.get(provider.lower())


def _models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _pipeline(df: pd.DataFrame, query: str, provider: str, model: str,
              save: Path, top_n: int | None) -> Path:
    """Zwraca ścieżkę do ZIP-a z rankingiem, topN i raportem."""
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine
    from .bibliometrics import full_report

    tcol, acol = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, tcol, acol)

    vecs = get_embeddings(df["combined_text"].tolist(),
                          provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]
    ranked = rank_by_cosine(qvec, vecs, df)

    p_all = save / "ranking.csv"
    ranked.to_csv(p_all, index=False)

    p_top = None
    if top_n:
        p_top = save / "topN.csv"
        ranked.head(top_n).to_csv(p_top, index=False)

    rep = save / "biblio_report.txt"
    full_report(ranked, path=rep, top_n=top_n)

    z = save / "embedslr_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(p_all, "ranking.csv")
        if p_top:
            zf.write(p_top, "topN.csv")
        zf.write(rep, "biblio_report.txt")
    return z


# ───────────── interactive Colab mode ───────────────────────────────────────
def _colab_ui(out_dir: Path):
    from google.colab import files  # type: ignore

    display(HTML(
        "<h3>EmbedSLR – interactive upload</h3>"
        "<ol><li><b>Browse</b> → wybierz CSV.</li>"
        "<li>Poczekaj na ✅ (plik wczytany).</li>"
        "<li>Wpisz odpowiedzi w konsoli.</li></ol>"
    ))
    up = files.upload()
    if not up:
        display(HTML("<b style='color:red'>No file uploaded – abort.</b>"))
        return
    name, data = next(iter(up.items()))
    df = pd.read_csv(io.BytesIO(data))
    display(HTML(f"✅ Loaded <code>{name}</code> ({len(df)} rows)<br>"))

    # ── prompts
    q = input("❓ Research query: ").strip()
    provs = list(_models())
    print("Providers:", provs)
    prov = input(f"Provider [default={provs[0]}]: ").strip() or provs[0]

    print("Models for", prov)
    for m in _models()[prov]:
        print("  •", m)
    mod = input("Model [ENTER=1st]: ").strip() or _models()[prov][0]

    n_raw = input("🔢 Top‑N publications for metrics [ENTER = all]: ").strip()
    top_n = int(n_raw) if n_raw else None

    key = input("API key (if required, ENTER skip): ").strip()
    if key and (ev := _env_var(prov)):
        os.environ[ev] = key

    print("⏳ Embedding & metrics…")
    zip_tmp = _pipeline(df, q, prov, mod, out_dir, top_n)

    dst = Path.cwd() / zip_tmp.name
    shutil.copy(zip_tmp, dst)
    print("✅ Done – downloading ZIP …")
    files.download(str(dst))          # auto‑download


# ───────────── CLI fallback ────────────────────────────────────────────────
def _cli(out: Path):
    print("== EmbedSLR CLI ==")
    csv_p = Path(input("CSV path: ").strip())
    df = pd.read_csv(csv_p)
    q = input("Query: ").strip()
    prov = input("Provider (sbert/openai/cohere/jina/nomic): ").strip() or "sbert"
    mod = input("Model [ENTER=default]: ").strip() or _models()[prov][0]
    n_raw = input("Top‑N [ENTER=all]: ").strip()
    top_n = int(n_raw) if n_raw else None
    key = input("API key [skip]: ").strip()
    if key and (ev := _env_var(prov)):
        os.environ[ev] = key
    z = _pipeline(df, q, prov, mod, out, top_n)
    print("ZIP saved:", z)


# ───────────── public ──────────────────────────────────────────────────────
def run(save_dir: str | os.PathLike | None = None):
    save_dir = Path(save_dir or tempfile.mkdtemp(prefix="embedslr_"))
    clear_output()
    (_colab_ui if IN_COLAB else _cli)(save_dir)
