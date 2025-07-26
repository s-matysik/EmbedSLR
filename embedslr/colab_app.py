# embedslr/colab_app.py
"""
Interactive wizard for Google Colab / Jupyter Lab
=================================================

1.  Upload a Scopus / Web‑of‑Science CSV.
2.  Enter the research question (query).
3.  Choose the embedding provider + model.
4.  (Optional) paste an API key if the provider requires it.
5.  Click **Start** – the pipeline runs and returns a ZIP containing:

    • ranking.csv – publications sorted by cosine distance  
    • biblio_report.txt – quick bibliometric diagnostics
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path

import ipywidgets as w
import pandas as pd
from IPython.display import FileLink, clear_output, display

from .bibliometrics import full_report
from .embeddings import get_embeddings, list_models
from .io import autodetect_columns, combine_title_abstract, read_csv
from .similarity import rank_by_cosine


# -----------------------------------------------------------------------------#
# Helper widgets
# -----------------------------------------------------------------------------#


def _upload_widget() -> w.FileUpload:
    """Return a FileUpload widget with a live status label."""
    status = w.Label(value="⏳ No file selected")
    upload = w.FileUpload(accept=".csv", multiple=False)

    def _parse(change):
        if upload.value:
            raw = next(iter(upload.value.values()))
            name = raw["metadata"]["name"]
            df = pd.read_csv(io.BytesIO(raw["content"]), low_memory=False)
            upload.df = df  # type: ignore[attr-defined]
            status.value = f"✅ Loaded *{name}* with {len(df)} records"

    upload.observe(_parse, names="value")
    box = w.VBox([w.HTML("<b>1. Upload CSV</b>"), upload, status])
    display(box)
    return upload


def _provider_widget() -> tuple[w.Dropdown, w.Dropdown]:
    """Return two linked dropdowns: provider ↔ model."""
    prov = w.Dropdown(options=list(list_models()), description="Provider")
    model = w.Dropdown(description="Model")

    def _refresh(_):
        model.options = list_models()[prov.value]
        model.value = model.options[0]

    prov.observe(_refresh, names="value")
    _refresh(None)
    return prov, model


def _set_api_key(provider: str, key: str) -> None:
    """Register *key* in os.environ for the given provider name."""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "nomic": "NOMIC_API_KEY",
        "jina": "JINA_API_KEY",
    }
    var = env_map.get(provider)
    if var:
        os.environ[var] = key


# -----------------------------------------------------------------------------#
# Public entry point
# -----------------------------------------------------------------------------#


def run(save_dir: str | os.PathLike | None = None) -> None:
    """
    Launch the interactive EmbedSLR wizard.

    Parameters
    ----------
    save_dir : str | Path, optional
        If provided, results are written there; otherwise a temporary directory
        is created.
    """
    save_dir = Path(save_dir or tempfile.mkdtemp(prefix="embedslr_"))

    # --- widgets ----------------------------------------------------------------
    upload = _upload_widget()
    query_box = w.Textarea(
        description="2. Query",
        placeholder="Describe your research problem…",
        layout=w.Layout(width="95%", height="80px"),
    )
    prov_dd, model_dd = _provider_widget()
    api_key = w.Password(description="API key")
    start_btn = w.Button(description="🟢 Start", button_style="success")
    output = w.Output()

    # --- callback ----------------------------------------------------------------
    def _on_start(_btn):
        output.clear_output()
        # --- validation ----------------------------------------------------------
        if not hasattr(upload, "df"):
            with output:
                print("❌ First upload a CSV file (step 1).")
            return
        if not query_box.value.strip():
            with output:
                print("❌ Enter a research query (step 2).")
            return

        df = upload.df  # type: ignore[attr-defined]

        with output:
            clear_output()
            print("▶︎ Processing …")

        # set env var if user pasted a key
        if api_key.value:
            _set_api_key(prov_dd.value, api_key.value)

        # --- preprocessing -------------------------------------------------------
        title_col, abs_col = autodetect_columns(df)
        df["combined_text"] = combine_title_abstract(df, title_col, abs_col)

        # --- embeddings ----------------------------------------------------------
        texts = df["combined_text"].tolist()
        embs = get_embeddings(texts, provider=prov_dd.value, model=model_dd.value)
        qvec = get_embeddings(
            [query_box.value], provider=prov_dd.value, model=model_dd.value
        )[0]

        # --- ranking & report ----------------------------------------------------
        ranked = rank_by_cosine(qvec, embs, df)
        csv_path = save_dir / "ranking.csv"
        ranked.to_csv(csv_path, index=False)

        rep_path = save_dir / "biblio_report.txt"
        full_report(ranked, path=rep_path)

        # bundle to ZIP for easy download
        zip_path = save_dir / "embedslr_results.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname="ranking.csv")
            zf.write(rep_path, arcname="biblio_report.txt")

        # --- final UI ------------------------------------------------------------
        with output:
            clear_output()
            display(w.HTML("<h4>✅ Done – download:</h4>"))
            display(FileLink(str(zip_path), result_html_prefix="📦 "))

    start_btn.on_click(_on_start)

    # --- layout -----------------------------------------------------------------
    display(
        w.VBox(
            [
                query_box,
                w.HBox([prov_dd, model_dd]),
                api_key,
                start_btn,
                output,
            ]
        )
    )
