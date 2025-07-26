"""
Interactive Colab wizard – upload CSV → select provider/model → get ZIP with results.
"""
from __future__ import annotations
import io, os, json, zipfile, tempfile, pandas as pd
import ipywidgets as w
from IPython.display import display, clear_output, FileLink

from .io import read_csv, autodetect_columns, combine_title_abstract
from .embeddings import get_embeddings, list_models
from .similarity import rank_by_cosine
from .bibliometrics import full_report


def _upload_widget():
    up = w.FileUpload(accept=".csv", multiple=False)
    display(w.HTML("<b>1. Prześlij plik CSV (Scopus/WoS)</b>"), up)

    def _parse(change):
        if up.value:
            raw = next(iter(up.value.values()))
            up.df = pd.read_csv(io.BytesIO(raw["content"]), low_memory=False)  # type: ignore

    up.observe(_parse, names="value")
    return up


def _provider_widget():
    prov = w.Dropdown(options=list(list_models()), description="Provider")
    model = w.Dropdown(description="Model")

    def _refresh(_):
        model.options = list_models()[prov.value]
        model.value = model.options[0]

    prov.observe(_refresh, names="value")
    _refresh(None)
    return prov, model


def run(save_dir: str | None = None):
    save_dir = save_dir or tempfile.mkdtemp(prefix="embedslr_")
    up = _upload_widget()
    qbox = w.Textarea(description="2. Zapytanie", layout=w.Layout(width="95%", height="70px"))
    prov_dd, model_dd = _provider_widget()
    key = w.Password(description="API key")
    btn = w.Button(description="🟢 Start", button_style="success")
    out = w.Output()

    def _work(_):
        clear_output(wait=True)
        with out:
            print(">>> Processing …")
        df = up.df  # type: ignore
        title, abstr = autodetect_columns(df)
        df["combined_text"] = combine_title_abstract(df, title, abstr)
        if key.value:
            var = {"openai": "OPENAI_API_KEY", "cohere": "COHERE_API_KEY",
                   "jina": "JINA_API_KEY", "nomic": "NOMIC_API_KEY"}[prov_dd.value]
            os.environ[var] = key.value

        embs = get_embeddings(df["combined_text"].tolist(),
                              provider=prov_dd.value, model=model_dd.value)
        qvec = get_embeddings([qbox.value], provider=prov_dd.value, model=model_dd.value)[0]
        ranked = rank_by_cosine(qvec, embs, df)

        csv_path = os.path.join(save_dir, "ranking.csv")
        ranked.to_csv(csv_path, index=False)
        rep_path = os.path.join(save_dir, "biblio_report.txt")
        full_report(ranked, path=rep_path)

        zip_path = os.path.join(save_dir, "embedslr_results.zip")
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(csv_path, "ranking.csv")
            z.write(rep_path, "biblio_report.txt")

        with out:
            clear_output()
            display(w.HTML("<h4>✅ Gotowe! Pobierz wyniki:</h4>"))
            display(FileLink(zip_path))

    btn.on_click(_work)
    display(w.VBox([qbox, w.HBox([prov_dd, model_dd]), key, btn, out]))
