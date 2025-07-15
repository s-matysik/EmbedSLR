# embedslr/embeddings/openai_api.py
from __future__ import annotations

import os
from typing import List

import openai
from openai import OpenAI

from .base import BaseEmbedder
from embedslr.utils.batching import chunk_list


class OpenAIEmbedder(BaseEmbedder):
    """
    Adapter do modelu OpenAI Embeddings (API>=1.0).

    Domyślny model: ``text-embedding-ada-002``.
    Klucz API pobierany z parametru konstruktora *lub* zmiennej
    środowiskowej ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "text-embedding-ada-002",
        batch_size: int = 100,  # limit wg dokumentacji OpenAI
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Brak klucza OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.batch_size = batch_size

    # ---------------------------------------------------------

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings: list[list[float]] = []
        for batch in chunk_list(texts, self.batch_size):
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            embeddings.extend([item.embedding for item in resp.data])
        return embeddings
