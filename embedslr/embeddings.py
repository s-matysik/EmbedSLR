"""
Unified façade for five providers of text embeddings.
"""
from __future__ import annotations
import requests
from typing import List

from sentence_transformers import SentenceTransformer
from openai import OpenAI
import cohere

from .utils import chunk_iterable, getenv_or_raise, progress

PROVIDERS = {"sbert", "openai", "cohere", "nomic", "jina"}


def list_models() -> dict[str, list[str]]:
    return {
        "sbert": ["sentence-transformers/all-mpnet-base-v2"],
        "openai": ["text-embedding-ada-002", "text-embedding-3-small"],
        "cohere": ["embed-english-v3.0"],
        "nomic": ["nomic-embed-text-v1.5"],
        "jina": ["jina-embeddings-v3"],
    }


def get_embeddings(
    texts: list[str], provider: str = "sbert", model: str | None = None, **kw
) -> list[list[float]]:
    fn = {
        "sbert": _sbert,
        "openai": _openai,
        "cohere": _cohere,
        "nomic": _nomic,
        "jina": _jina,
    }.get(provider.lower())
    if fn is None:
        raise ValueError(f"Unknown provider {provider}")
    return fn(texts, model=model, **kw)


# ---------- provider‑specific ------------------------------------------------- #


def _sbert(texts, model=None, **_):
    mname = model or "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(mname)
    with progress("SBERT", total=len(texts)):
        return model.encode(texts, show_progress_bar=False).tolist()


def _openai(texts, model=None, **_):
    client = OpenAI(api_key=getenv_or_raise("OPENAI_API_KEY", "OpenAI"))
    mname = model or "text-embedding-ada-002"
    out: list[list[float]] = []
    with progress("OpenAI"):
        for batch in chunk_iterable(texts, 1000):
            res = client.embeddings.create(model=mname, input=batch)
            out.extend([d.embedding for d in res.data])
    return out


def _cohere(texts, model=None, **kw):
    co = cohere.Client(getenv_or_raise("COHERE_API_KEY", "Cohere"))
    mname = model or "embed-english-v3.0"
    input_type = kw.get("input_type", "classification")
    embs: list[list[float]] = []
    with progress("Cohere"):
        for batch in chunk_iterable(texts, 96):
            res = co.embed(
                texts=batch,
                model=mname,
                input_type=input_type,
                embedding_types=["float"],
            )
            embs.extend(res.embeddings.float)
    return embs


def _nomic(texts, model=None, **kw):
    api_key = getenv_or_raise("NOMIC_API_KEY", "Nomic")
    url = "https://api-atlas.nomic.ai/v1/embedding/text"
    base = {
        "model": model or "nomic-embed-text-v1.5",
        "task_type": kw.get("task_type", "search_document"),
        "long_text_mode": kw.get("long_text_mode", "truncate"),
    }
    embs: list[list[float]] = []
    with progress("Nomic"):
        for batch in chunk_iterable(texts, 100):
            payload = base | {"texts": batch}
            r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"})
            r.raise_for_status()
            embs.extend(r.json()["embeddings"])
    return embs


def _jina(texts, model=None, **kw):
    api_key = getenv_or_raise("JINA_API_KEY", "Jina AI")
    url = "https://api.jina.ai/v1/embeddings"
    mname = model or "jina-embeddings-v3"
    embs: list[list[float]] = []
    with progress("Jina AI"):
        for batch in chunk_iterable(texts, 100):
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": mname,
                    "task": kw.get("task", "text-matching"),
                    "dimensions": kw.get("dimensions", 1024),
                    "embedding_type": kw.get("embedding_type", "float"),
                    "input": [{"text": t} for t in batch],
                },
                timeout=60,
            )
            r.raise_for_status()
            embs.extend([item["embedding"] for item in r.json()["data"]])
    return embs
