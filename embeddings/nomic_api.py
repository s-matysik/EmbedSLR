# embedslr/embeddings/nomic_api.py
from __future__ import annotations

import os
from typing import List

import requests
from .base import BaseEmbedder
from embedslr.utils.batching import chunk_list


class NomicEmbedder(BaseEmbedder):
    """
    Adapter do modelu Atlas / Nomic Embed (`nomic-embed-text-v1.5`).

    Komunikuje się bezpośrednio po REST API.
    """

    API_URL = "https://api-atlas.nomic.ai/v1/embedding/text"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "nomic-embed-text-v1.5",
        task_type: str = "search_document",
        long_text_mode: str = "truncate",
        batch_size: int = 100,
    ) -> None:
        self.api_key = api_key or os.getenv("NOMIC_API_KEY")
        if not self.api_key:
            raise ValueError("Brak klucza NOMIC_API_KEY")
        self.model_name = model_name
        self.task_type = task_type
        self.long_text_mode = long_text_mode
        self.batch_size = batch_size

    # ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def encode(self, texts: List[str]) -> List[List[float]]:
        result: list[list[float]] = []
        for batch in chunk_list(texts, self.batch_size):
            payload = {
                "texts": batch,
                "model": self.model_name,
                "task_type": self.task_type,
                "long_text_mode": self.long_text_mode,
            }
            resp = requests.post(self.API_URL, headers=self._headers(), json=payload)
            resp.raise_for_status()
            result.extend(resp.json()["embeddings"])
        return result
