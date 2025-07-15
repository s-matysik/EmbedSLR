# embedslr/embeddings/jina_api.py
from __future__ import annotations

import os
from typing import List

import requests
from .base import BaseEmbedder
from embedslr.utils.batching import chunk_list


class JinaEmbedder(BaseEmbedder):
    """
    Adapter do `jina.ai` (`/v1/embeddings`).

    Domyślne parametry odpowiadają dokumentacji v3.
    """

    API_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "jina-embeddings-v3",
        task: str = "text-matching",
        dimensions: int = 1024,
        batch_size: int = 100,
    ) -> None:
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Brak klucza JINA_API_KEY")
        self.model_name = model_name
        self.task = task
        self.dimensions = dimensions
        self.batch_size = batch_size

    # ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings: list[list[float]] = []
        for batch in chunk_list(texts, self.batch_size):
            payload = {
                "model": self.model_name,
                "task": self.task,
                "dimensions": self.dimensions,
                "embedding_type": "float",
                "late_chunking": False,
                "input": [{"text": t} for t in batch],
            }
            resp = requests.post(self.API_URL, headers=self._headers(), json=payload)
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"Jina API error: {data['error']}")
            embeddings.extend(item["embedding"] for item in data["data"])
        return embeddings
