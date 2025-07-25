
"""
OpenAI / Azure OpenAI backend dla EmbedSLR.
"""

from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import EmbeddingBackend


class OpenAIEmbedder(EmbeddingBackend):
    """
    Generuje wektory używając endpointu *embeddings.create*.

    Parameters
    ----------
    model:
        domyślnie "text-embedding-3-small"; możesz podać "text-embedding-3-large"
        lub dowolny model kompatybilny z API V2.
    client_kwargs:
        dodatkowe argumenty przekazywane do `openai.OpenAI(...)`
        (np. api_key, base_url).
    """

    def __init__(self, model: str = "text-embedding-3-small", **client_kwargs):
        self._model = model
        self._client = OpenAI(**client_kwargs)

    # ————— ochrona przed sporadycznym 5xx / RateLimit ———————————
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _call_api(self, texts: List[str]):
        return self._client.embeddings.create(model=self._model, input=texts)

    # ————————————————————————————————————————————————————————————————
    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self._call_api(texts)
        # API V2 zwraca listę obiektów z polem .embedding
        return [d.embedding for d in response.data]
