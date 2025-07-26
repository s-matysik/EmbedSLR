# embedslr/colab_app.py  ——  version 0.2.3
#
# działa w Google Colab *oraz* w klasycznym JupyterLab.
# ❶ próbuje użyć ipywidgets;
# ❷ jeżeli widgety nie są obsługiwane / zablokowane, przechodzi w tryb Fallback
#    (upload pliku + pola tekstowe + przycisk HTML).
#
# © 2025 EmbedSLR team – MIT License
# ---------------------------------------------------------------------------

from __future__ import annotations

import io
import os
import sys
import zipfile
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
from IPython.display import display, clear_output, HTML, FileLink

# --- probe for Colab environment & widget availability ---------------------
IN_COLAB = "google.colab" in sys.modules
WIDGETS_OK = False
try:
    import ipywidgets as W
    # Colab 2024–07 domyślnie ma widgets 7.x. Jeśli użytkownik zainstalował 8.x,
    # działa, o ile wywołano enable_custom_widget_manager (niżej).
    from google.colab import output as _col_out  # type: ignore
    _col_out.enable_custom_widget_manager()
    WIDGETS_OK = True
except Exception:
    WIDGETS_OK = False


# ------- generic helpers ----------------------------------------------------
def _set_api_key(provider: str, key: str) -> None:
    env_map = {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "jina": "JINA_API_KEY",
        "nomic": "NOMIC_API_KEY",
    }
    var = env_map.get(provider.lower())
    if var:
        os.environ[var] = key


def _list_models() -> dict[str, List[str]]:
    from .embeddings import list_models  # lazy import
    return list_models()


# ------- PIPELINE core  -----------------------------------------------------
def _run_pipeline(
    df: pd.DataFrame,
    query: str,
    provider: str,
    model: str,
    save_dir: Path,
) -> Path:
    """Generuje ranking + raport ➜ pakuje ZIP i zwraca ścieżkę."""
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine
    from .bibliometrics import full_report

    title, abstr = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, title, abstr)

    texts = df["combined_text"].tolist()
    vecs = get_embeddings(texts, provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]

    ranked = rank_by_cosine(qvec, vecs, df)
    csv_path = save_dir / "ranking.csv"
    ranked.to_csv(csv_path, index=False)

    rep_path = save_dir / "biblio_report.txt"
    full_report(ranked, path=rep_path)

    zip_path = save_dir / "embedslr_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="ranking.csv")
        zf.write(rep_path, arcname="biblio_report.txt")
    return zip_path


# ---------------------------------------------------------------------------#
#  WIDGET MODE  (preferred)
# ---------------------------------------------------------------------------#
def _widget_mode(save_dir: Path):
    import ipywidgets as W

    # 1 · UPLOAD
    upload = W.FileUpload(accept=".csv", multiple=False)
    status = W.HTML("<i>No file selected</i>")

    def _on_upload(change):
        if upload.value:
            raw = next(iter(upload.value.values()))
            upload.df = pd.read_csv(io.BytesIO(raw["content"]))  # type: ignore
            status.value = f"✅ {raw['metadata']['name']}  — {len(upload.df)} rows"

    upload.observe(_on_upload, names="value")
    display(W.VBox([W.HTML("<b>1. Upload CSV</b>"), upload, status]))

    # 2 · QUERY
    query = W.Textarea(
        description="2. Query",
        placeholder="Type your research problem…",
        layout=W.Layout(width="95%", height="70px"),
    )

    # 3 · PROVIDER + MODEL
    prov_dd = W.Dropdown(options=list(_list_models()), description="Provider")
    model_dd = W.Dropdown(description="Model")

    def _refresh(_):
        model_dd.options = _list_models()[prov_dd.value]
        model_dd.value = model_dd.options[0]

    prov_dd.observe(_refresh, names="value")
    _refresh(None)

    api_key = W.Password(description="API key")
    start = W.Button(description="🟢 Start", button_style="success")
    out = W.Output()

    def _start(_btn):
        out.clear_output()
        # basic validation
        if not hasattr(upload, "df"):
            with out:
                print("❌ Upload a CSV first.")
            return
        if not query.value.strip():
            with out:
                print("❌ Enter a query.")
            return

        if api_key.value:
            _set_api_key(prov_dd.value, api_key.value)

        with out:
            clear_output()
            print("⏳ Processing …")

        zip_path = _run_pipeline(
            upload.df,  # type: ignore
            query.value,
            prov_dd.value,
            model_dd.value,
            save_dir,
        )
        with out:
            clear_output()
            display(HTML("<h4>✅ Done — download:</h4>"))
            display(FileLink(str(zip_path), result_html_prefix="📦 "))

    start.on_click(_start)

    display(
        W.VBox(
            [
                query,
                W.HBox([prov_dd, model_dd]),
                api_key,
                start,
                out,
            ]
        )
    )


# ---------------------------------------------------------------------------#
#  FALLBACK MODE  (no ipywidgets)
# ---------------------------------------------------------------------------#
def _fallback_mode(save_dir: Path):
    from google.colab import files  # type: ignore

    display(
        HTML(
            """
    <h3>EmbedSLR – fallback mode (no ipywidgets)</h3>
    <ol>
      <li>Click <b>Browse</b> to upload a Scopus/WoS CSV.</li>
      <li>When upload finishes, type your research query below.</li>
      <li>Select provider, paste API key (if needed) and click <em>Run</em>.</li>
    </ol>
    """
        )
    )

    # 1. upload
    uploaded = files.upload()
    if not uploaded:
        display(HTML("<b style='color:red'>No file uploaded – aborting.</b>"))
        return
    fname = list(uploaded.keys())[0]
    df = pd.read_csv(fname)
    display(HTML(f"✅ Loaded <code>{fname}</code> with {len(df)} rows.<br>"))

    # 2. query
    query = input("❓ Research query: ").strip()
    if not query:
        print("Abort – query cannot be empty.")
        return

    # 3. provider + model
    providers = list(_list_models())
    print(f"Available providers: {providers}")
    provider = input(f"Provider [{providers[0]}]: ").strip() or providers[0]
    model = input("Model (leave blank = default): ").strip() or _list_models()[provider][0]
    key = input("API key (if required, else leave blank): ").strip()
    if key:
        _set_api_key(provider, key)

    print("⏳ Running EmbedSLR …")
    zip_path = _run_pipeline(df, query, provider, model, save_dir)
    print(f"✅ Done. Download: {zip_path}")


# ---------------------------------------------------------------------------#
#  PUBLIC ENTRY
# ---------------------------------------------------------------------------#
def run(save_dir: str | os.PathLike | None = None):
    """
    Launch interactive wizard (ipywidgets) or fallback CLI if widgets unavailable.
    """
    save_dir = Path(save_dir or tempfile.mkdtemp(prefix="embedslr_"))
    if WIDGETS_OK:
        _widget_mode(save_dir)
    else:
        print("⚠️ ipywidgets are unavailable in this environment – "
              "switching to fallback mode.")
        _fallback_mode(save_dir)
