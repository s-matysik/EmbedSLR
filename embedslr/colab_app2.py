from __future__ import annotations
import io, os, sys, tempfile, zipfile, shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from IPython.display import HTML, clear_output, display

IN_COLAB = "google.colab" in sys.modules


# ───────── helpers ─────────────────────────────────────────────────────────
def _env_var(p: str) -> str | None:
    return {"openai": "OPENAI_API_KEY", "cohere": "COHERE_API_KEY",
            "jina": "JINA_API_KEY", "nomic": "NOMIC_API_KEY"}.get(p.lower())


def _models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _ensure_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy Parsed_References / Author Keywords jeżeli brak."""
    if "Parsed_References" not in df.columns:
        if "References" in df.columns:
            df["Parsed_References"] = df["References"].fillna("").apply(
                lambda x: {r.strip() for r in x.split(");") if r.strip()}
            )
        else:
            df["Parsed_References"] = [set()] * len(df)

    if "Author Keywords" not in df.columns:
        df["Author Keywords"] = ""

    # spójny Title
    if "Title" not in df.columns:
        if "Article Title" in df.columns:
            df["Title"] = df["Article Title"]
        else:
            df["Title"] = [f"Paper_{i}" for i in range(len(df))]
    return df


def _pipeline(df: pd.DataFrame, query: str, provider: str, model: str,
              out: Path, top_n: int | None) -> Path:
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine

    # Import tylko potrzebnych wskaźników (A, B, F) + wspólne statystyki
    try:
        from .bibliometric import indicator_a, indicator_b, indicator_f, _prepare_stats  # type: ignore
    except ImportError:
        from .bibliometrics import indicator_a, indicator_b, indicator_f, _prepare_stats  # type: ignore

    df = _ensure_aux_columns(df.copy())
    tcol, acol = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, tcol, acol)

    vecs = get_embeddings(df["combined_text"].tolist(),
                          provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]
    ranked = rank_by_cosine(qvec, vecs, df)

    # Zapis rankingów
    p_all = out / "ranking.csv"
    ranked.to_csv(p_all, index=False)

    p_top = None
    if top_n:
        p_top = out / "topN.csv"
        ranked.head(top_n).to_csv(p_top, index=False)

    # ── TYLKO 3 wybrane wskaźniki (A, B, F) ────────────────────────────────
    sub = ranked.head(top_n) if top_n else ranked
    stats = _prepare_stats(sub)  # wspólne liczenie aggregatów (1 przebieg)

    val_a = indicator_a(sub, _stats=stats)
    val_b = indicator_b(sub, _stats=stats)
    val_f = indicator_f(sub, _stats=stats)

    rep = out / "biblio_report.txt"
    lines = [
        "==== BIBLIOMETRIC SUMMARY (A, B, F) ====",
        f"{'A – Avg shared refs / pair':32s}: {val_a:.4f}",
        f"{'B – Avg shared kws / pair':32s}: {val_b:.4f}",
        f"{'F – Pairs ≥1 common kw':32s}: {val_f}",
    ]
    with open(rep, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # ZIP z wynikami
    zf = out / "embedslr_results.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p_all, "ranking.csv")
        if p_top:
            z.write(p_top, "topN.csv")
        z.write(rep, "biblio_report.txt")
    return zf


# ───────── interactive Colab ───────────────────────────────────────────────
def _colab_ui(out_dir: Path):
    from google.colab import files  # type: ignore

    display(HTML(
        "<h3>EmbedSLR – interactive upload</h3>"
        "<ol><li><b>Browse</b> → CSV</li><li>Wait for ✅</li>"
        "<li>Answer prompts in console</li></ol>"
    ))
    up = files.upload()
    if not up:
        display(HTML("<b style='color:red'>abort – no file</b>")); return
    name, data = next(iter(up.items()))
    df = pd.read_csv(io.BytesIO(data))
    display(HTML(f"✅ Loaded <code>{name}</code> ({len(df)} rows)<br>"))

    q = input("❓ Research query: ").strip()
    provs = list(_models())
    print("Providers:", provs)
    prov = input(f"Provider [default={provs[0]}]: ").strip() or provs[0]

    print("Models for", prov)
    for m in _models()[prov]:
        print("  •", m)
    mod = input("Model [ENTER=1st]: ").strip() or _models()[prov][0]

    n_raw = input("🔢 Top‑N for metrics [ENTER=all]: ").strip()
    top_n = int(n_raw) if n_raw else None

    key = input("API key (ENTER skip): ").strip()
    if key and (ev := _env_var(prov)):
        os.environ[ev] = key

    print("⏳ Computing …")
    zip_tmp = _pipeline(df, q, prov, mod, out_dir, top_n)

    dst = Path.cwd() / zip_tmp.name
    shutil.copy(zip_tmp, dst)
    print("✅ Finished – downloading ZIP")
    files.download(str(dst))


# ───────── CLI fallback ────────────────────────────────────────────────────
def _cli(out_dir: Path):
    print("== EmbedSLR CLI ==")
    csv_p = Path(input("CSV path: ").strip())
    df = pd.read_csv(csv_p)
    q = input("Query: ").strip()
    prov = input("Provider [sbert]: ").strip() or "sbert"
    mod = input("Model [ENTER=default]: ").strip() or _models()[prov][0]
    n_raw = input("Top‑N [ENTER=all]: ").strip()
    top_n = int(n_raw) if n_raw else None
    key = input("API key [skip]: ").strip()
    if key and (ev := _env_var(prov)):
        os.environ[ev] = key
    z = _pipeline(df, q, prov, mod, out_dir, top_n)
    print("ZIP saved:", z)


# ───────── public ─────────────────────────────────────────────────────────
def run(save_dir: str | os.PathLike | None = None):
    save_dir = Path(save_dir or tempfile.mkdtemp(prefix="embedslr_"))
    clear_output()
    (_colab_ui if IN_COLAB else _cli)(save_dir)
