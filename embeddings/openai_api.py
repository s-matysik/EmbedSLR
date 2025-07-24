# embedslr/embeddings/openai_api.py
from __future__ import annotations

import os
from typing import List, Sequence

# ──────────────────────────────────────────────────────────
#  Wspieramy jednocześnie  openai >= 1.0  *i* < 1.0
try:                       # nowe API (klasa OpenAI)
    from openai import OpenAI
    _HAS_NEW_OPENAI = True
except ImportError:        # stare API (moduł openai w wersji 0.27)
    import openai          # type: ignore
    _HAS_NEW_OPENAI = False

from .base import BaseEmbedder
from embedslr.utils.batching import chunk_list


class OpenAIEmbedder(BaseEmbedder):
    """
    Embeddings‑backend OpenAI.

    * działa z ``openai>=1.0`` *oraz* ``openai<1.0``,
    * domyślny model : ``text-embedding-ada-002``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "text-embedding-ada-002",
        batch_size: int = 100,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Brak klucza API – podaj go w konstruktorze albo ustaw "
                "zmienną środowiskową OPENAI_API_KEY."
            )
        self.model_name = model_name
        self.batch_size = batch_size

        # konfiguracja klienta zależnie od wersji SDK
        if _HAS_NEW_OPENAI:
            self.client = OpenAI(api_key=self.api_key)
            self._embed_fn = self._embed_new
        else:
            openai.api_key = self.api_key            # type: ignore[attr-defined]
            self._embed_fn = self._embed_old

    # ────────────────────────────────────────────────────
    # API wymagane przez BaseEmbedder
    def embed(self, texts: Sequence[str]) -> List[List[float]]:   # type: ignore[override]
        return self._embed_fn(list(texts))

    # ────────────────────────────────────────────────────
    # Implementacje prywatne
    def _embed_new(self, texts: List[str]) -> List[List[float]]:
        vectors: list[list[float]] = []
        for batch in chunk_list(texts, self.batch_size):
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend([item.embedding for item in resp.data])
        return vectors

    def _embed_old(self, texts: List[str]) -> List[List[float]]:
        import openai                           # type: ignore
        vectors: list[list[float]] = []
        for batch in chunk_list(texts, self.batch_size):
            resp = openai.Embedding.create(model=self.model_name, input=batch)
            vectors.extend([item["embedding"] for item in resp["data"]])
        return vectors
