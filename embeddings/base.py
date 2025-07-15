from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):
    """Abstrakcyjny interfejs wszystkich silników embeddingowych."""

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Zwraca listę wektorów (list[float]) dla podanej listy tekstów."""
        raise NotImplementedError
