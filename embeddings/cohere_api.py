# embedslr/embeddings/cohere_api.py
from __future__ import annotations

import os
from typing import List

import cohere
from .base import BaseEmbedder
from embedslr.utils.batching import chunk_list


class CohereEmbedder(BaseEmbedder):
    """
    Adapter do modelu Cohere v3 embed-*.

    Domyślny model: ``embed-english-v3.0``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "embed-english-v3.0",
        input_type: str = "classification",
        batch_size: int = 96,  # limit Cohere
    ) -> None:
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Brak klucza COHERE_API_KEY")
        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model_name = model_name
        self.input_type = input_type
        self.batch_size = batch_size

    # ---------------------------------------------------------

    def encode(self, texts: List[str]) -> List[List[float]]:
        out: list[list[float]] = []
        for batch in chunk_list(texts, self.batch_size):
            resp = self.client.embed(
                texts=batch,
                model=self.model_name,
                input_type=self.input_type,
                embedding_types=["float"],
            )
            out.extend(resp.embeddings.float)
        return out
