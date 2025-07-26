# embedslr/colab_app.py   ——  v0.3.1  ——  MIT
from __future__ import annotations
import io, os, shutil, sys, zipfile, tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
from IPython.display import HTML, clear_output, display

IN_COLAB = "google.colab" in sys.modules


# ─── utils ────────────────────────────────────────────────────────────────
def _set_api_key(provider: str, key: str) -> None:
    env = {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "jina":   "JINA_API_KEY",
        "nomic":  "NOMIC_API_KEY",
    }.get(provider.lower())
    if env:
        os.environ[env] = key


def _models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _pipeline(df: pd.DataFrame, query: str, provider: str,
              model: str, save_dir: Path) -> Path:
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

    csv_p = save_dir / "ranking.csv"
    ranked.to_csv(csv_p, index=False)
    rep_p = save_dir / "biblio_report.txt"
    full_report(ranked, path=rep_p)

    zip_p = save_dir / "embedslr_results.zip"
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_p, arcname="ranking.csv")
        zf.write(rep_p, arcname="biblio_report.txt")
    return zip_p


# ─── Colab interactive mode (widget‑free) ─────────────────────────────────
def _colab_mode(save_dir: Path):
    from google.colab import files  # type: ignore

    display(HTML(
        "<h3>EmbedSLR – interactive upload</h3>"
        "<ol><li><b>Browse</b> → wybierz plik CSV.</li>"
        "<li>Poczekaj na ✅.</li>"
        "<li>Odpowiedz na pytania w konsoli.</li></ol>"
    ))

    uploaded = files.upload()        # natywny dialog Colaba
    if not uploaded:
        display(HTML("<b style='color:red'>Brak pliku – przerywam.</b>"))
        return
    fname, data = next(iter(uploaded.items()))
    df = pd.read_csv(io.BytesIO(data))
    display(HTML(f"✅ Wczytano <code>{fname}</code> ({len(df)} rekordów)<br>"))

    query = input("❓ Query: ").strip()
    provs = list(_models())
    provider = input(f"Provider {provs} [default={provs[0]}]: ").strip() or provs[0]
    model = input("Model (ENTER=default): ").strip() or _models()[provider][0]
    key = input("API key (ENTER=pomiń): ").strip()
    if key:
        _set_api_key(provider, key)

    print("⏳ Running EmbedSLR …")
    zip_tmp = _pipeline(df, query, provider, model, save_dir)

    # ►► przenosimy ZIP do /content, aby był widoczny; potem download
    zip_name = Path(zip_tmp).name
    zip_final = Path.cwd() / zip_name
    shutil.copy(zip_tmp, zip_final)

    print(f"✅ Done. File saved as {zip_name}")
    files.download(str(zip_final))     # wywołuje okno „Save file”


# ─── CLI fallback (zwykły Jupyter) ────────────────────────────────────────
def _cli_mode(save_dir: Path):
    print("== EmbedSLR CLI ==")
    csv_path = input("CSV path: ").strip()
    df = pd.read_csv(csv_path)
    query = input("Query: ").strip()
    prov = input("Provider (sbert/openai/cohere/nomic/jina): ").strip() or "sbert"
    model = input("Model (ENTER=default): ").strip() or _models()[prov][0]
    key = input("API key (ENTER=skip): ").strip()
    if key:
        _set_api_key(prov, key)

    zip_p = _pipeline(df, query, prov, model, save_dir)
    print(f"ZIP saved: {zip_p}")


# ─── public API ───────────────────────────────────────────────────────────
def run(save_dir: str | os.PathLike | None = None) -> None:
    """Start Colab interactive upload or simple CLI."""
    save_dir = Path(save_dir or tempfile.mkdtemp(prefix="embedslr_"))
    clear_output()
    if IN_COLAB:
        _colab_mode(save_dir)
    else:
        _cli_mode(save_dir)
