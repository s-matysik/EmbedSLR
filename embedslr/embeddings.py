"""
embedslr.embeddings 

"""
from __future__ import annotations
import functools, os, time
from typing import Dict, List

import requests
from sentence_transformers import SentenceTransformer
from openai import OpenAI, APIConnectionError
import cohere

from .utils import chunk_iterable, getenv_or_raise, progress

# ────────────────────────────────────────────────────────────────────────────
_PROVIDERS = {"sbert", "openai", "cohere", "nomic", "jina"}

_STATIC_MODELS: Dict[str, List[str]] = {
    "sbert": [
        # fast / medium-sized
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        # multilingual & domain
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "sentence-transformers/average_word_embeddings_glove.6B.300d",
    ],
    "openai": [  # fallback when no key is provided
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ],
    "cohere": [
        "embed-english-v3.0",
        "embed-english-light-v3.0",
        "embed-multilingual-v3.0",
        "embed-multilingual-light-v3.0",
    ],
    "nomic": ["nomic-embed-text-v1", "nomic-embed-text-v1.5"],
    "jina":  ["jina-embeddings-v3"],
}


# ─────────────── dynamic OpenAI discovery (cache 6 h) ──────────────────────
@functools.lru_cache(maxsize=1)
def _openai_model_list() -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _STATIC_MODELS["openai"]

    client = OpenAI(api_key=api_key)
    try:
        models = client.models.list().data
    except APIConnectionError:
        # no internet or timeout – fallback to static list
        return _STATIC_MODELS["openai"]

    names = [m.id for m in models if m.id.startswith("text-embedding")]
    # sort by “family/m-size” → nicer for the eye
    names.sort()
    # add fallback in case OpenAI removed the old ada in the meantime
    return names or _STATIC_MODELS["openai"]


def list_models() -> Dict[str, List[str]]:
    """Returns a dictionary of available models (for OpenAI refreshed dynamically)."""
    out = _STATIC_MODELS.copy()
    out["openai"] = _openai_model_list()
    return out


# ────────────────────────────────────────────────────────────────────────────
def get_embeddings(
    texts: List[str],
    provider: str = "sbert",
    model: str | None = None,
    strict: bool = False,
    **kw,
) -> List[List[float]]:
    """
    Retrieves embeddings for the list *texts*.

    *provider*  – one of {`sbert`,`openai`,`cohere`,`nomic`,`jina`}
    *model*     – model name (can be arbitrary); if `None`, the first model
                  from the list returned by `list_models()` is used.
    *strict*    – when True and *model* is not on the list → ValueError.
    **kw        – parameters specific to the given provider (see below).
    """
    provider = provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'.")

    known = list_models()[provider]
    if model is None:
        model = known[0]
    elif strict and model not in known:
        raise ValueError(f"Model '{model}' not recognised for provider '{provider}'")

    fn = {
        "sbert": _embed_sbert,
        "openai": _embed_openai,
        "cohere": _embed_cohere,
        "nomic": _embed_nomic,
        "jina": _embed_jina,
    }[provider]

    return fn(texts, model=model, **kw)


# ───────────────────── provider-specific implementations ─────────────────────
def _embed_sbert(texts, *, model, **_):
    st = SentenceTransformer(model)
    with progress("SBERT", total=len(texts)):
        return st.encode(texts, show_progress_bar=False).tolist()


def _embed_openai(texts, *, model, **_):
    client = OpenAI(api_key=getenv_or_raise("OPENAI_API_KEY", "OpenAI"))
    res: List[List[float]] = []
    with progress("OpenAI"):
        for batch in chunk_iterable(texts, 1000):
            data = client.embeddings.create(model=model, input=batch).data
            res.extend([d.embedding for d in data])
    return res


def _embed_cohere(texts, *, model, input_type="classification",
                  embedding_types: List[str] | None = None, **_):
    co = cohere.Client(getenv_or_raise("COHERE_API_KEY", "Cohere"))
    et = embedding_types or ["float"]
    embs: List[List[float]] = []
    with progress("Cohere"):
        for batch in chunk_iterable(texts, 96):
            r = co.embed(texts=batch, model=model,
                         input_type=input_type, embedding_types=et)
            embs.extend(r.embeddings.float if "float" in et else r.embeddings.base64)
    return embs


def _embed_nomic(texts, *, model, task_type="search_document",
                 long_text_mode="truncate", dimensionality=None, **_):
    key = getenv_or_raise("NOMIC_API_KEY", "Nomic")
    url = "https://api-atlas.nomic.ai/v1/embedding/text"

    base = {"model": model, "task_type": task_type,
            "long_text_mode": long_text_mode}
    if dimensionality is not None:
        base["dimensionality"] = dimensionality

    embs: List[List[float]] = []
    with progress("Nomic"):
        for batch in chunk_iterable(texts, 100):
            r = requests.post(url, json=base | {"texts": batch},
                              headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            embs.extend(r.json()["embeddings"])
    return embs


def _embed_jina(texts, *, model, task="text-matching", dimensions=1024,
                embedding_type="float", **_):
    key = getenv_or_raise("JINA_API_KEY", "Jina AI")
    url = "https://api.jina.ai/v1/embeddings"

    embs: List[List[float]] = []
    with progress("Jina AI"):
        for batch in chunk_iterable(texts, 100):
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={
                    "model": model,
                    "task": task,
                    "dimensions": dimensions,
                    "embedding_type": embedding_type,
                    "input": [{"text": t} for t in batch],
                },
                timeout=60,
            )
            r.raise_for_status()
            embs.extend([item["embedding"] for item in r.json()["data"]])
    return embs
