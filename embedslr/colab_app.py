# embedslr/colab_app.py
"""
EmbedSLR – Google Colab wizard
-------------------------------
Upload  ➜  Query  ➜  Provider+Model  ➜  (API key)  ➜  Start  ➜  ZIP result.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import ipywidgets as w
from IPython.display import display, clear_output, FileLink

# --- try to enable third‑party widgets in new Colab UI -----------------------
try:
    from google.colab import output  # type: ignore

    output.enable_custom_widget_manager()
except Exception:  # noqa: BLE001
    # Notebook may not be running inside Colab or widgets already enabled
    pass


from .embeddings import get_embeddings, list_models
from .io import autodetect_columns, combine_title_abstract
from .similarity import rank_by_cosine
from .bibliometrics import full_report


# --------------------------------------------------------------------------- #
# Helper UI fragments
# --------------------------------------------------------------------------- #
def _upload_widget() -> w.FileUpload:
    status = w.HTML("<i>Awaiting file…</i>")
    upload = w.FileUpload(accept=".csv", multiple=False)

    def _on_upload(change):
        if upload.value:
            raw = next(iter(upload.value.values()))
            df = pd.read_csv(io.BytesIO(raw["content"]), low_memory=False)
            upload.df = df  # type: ignore[attr-defined]
            status.value = (
                f"✅ Loaded <b>{raw['metadata']['name']}</b> "
                f"(<code>{len(df)}</code> rows)"
            )

    upload.observe(_on_upload, names="value")
    box = w.VBox([w.HTML("<b>1 · Upload CSV (Scopus / WoS)</b>"), upload, status])
    display(box)
    return upload


def _provider_widgets() -> tuple[w.Dropdown, w.Dropdown]:
    prov = w.Dropdown(options=list(list_models()), description="Provider")
    model = w.Dropdown(description="Model")

    def _refresh(_):  # noqa: ANN001
        model.options = list_models()[prov.value]
        model.value = model.options[0]

    prov.observe(_refresh, names="value")
    _refresh(None)
    return prov, model


def _set_api_key(provider: str, key: str):
    mapping = {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "nomic": "NOMIC_API_KEY",
        "jina": "JINA_API_KEY",
    }
    var = mapping.get(provider)
    if var:
        os.environ[var] = key


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #
def run(save_dir: str | os.PathLike | None = None):
    """
    Launch the interactive wizard. Works in Google Colab and plain Jupyter.
    """
    if "google.colab" in sys.modules:
        print(
            "🔧  If you see a grey banner “third‑party Jupyter widgets” "
            "on the left, click ‘Enable’. Otherwise widgets will not render."
        )

    save_dir = Path(save_dir or tempfile.mkdtemp(prefix="embedslr_"))

    upload = _upload_widget()
    query = w.Textarea(
        description="2 · Query",
        placeholder="Type your research problem …",
        layout=w.Layout(width="95%", height="80px"),
    )
    prov_dd, model_dd = _provider_widgets()
    api_key = w.Password(description="API key")
    start = w.Button(description="🟢 Start", button_style="success")
    out = w.Output()

    # ------------------------------------------------ callback -------------
    def _start(_btn):
        out.clear_output()
        if not hasattr(upload, "df"):
            with out:
                print("❌ Upload a CSV first (step 1).")
            return
        if not query.value.strip():
            with out:
                print("❌ Fill in the research query (step 2).")
            return

        df = upload.df  # type: ignore[attr-defined]
        if api_key.value:
            _set_api_key(prov_dd.value, api_key.value)

        with out:
            clear_output()
            print("▶︎ Running EmbedSLR … please wait")

        title, abstr = autodetect_columns(df)
        df["combined_text"] = combine_title_abstract(df, title, abstr)

        embs = get_embeddings(
            df["combined_text"].tolist(), provider=prov_dd.value, model=model_dd.value
        )
        qvec = get_embeddings(
            [query.value], provider=prov_dd.value, model=model_dd.value
        )[0]

        ranked = rank_by_cosine(qvec, embs, df)
        csv_path = save_dir / "ranking.csv"
        ranked.to_csv(csv_path, index=False)
        rep_path = save_dir / "biblio_report.txt"
        full_report(ranked, path=rep_path)

        zip_path = save_dir / "embedslr_results.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname="ranking.csv")
            zf.write(rep_path, arcname="biblio_report.txt")

        with out:
            clear_output()
            display(w.HTML("<h4>✅ Finished — download your results:</h4>"))
            display(FileLink(str(zip_path), result_html_prefix="📦 "))

    start.on_click(_start)

    # ------------------------------------------------ layout ---------------
    display(
        w.VBox(
            [
                query,
                w.HBox([prov_dd, model_dd]),
                api_key,
                start,
                out,
            ]
        )
    )
