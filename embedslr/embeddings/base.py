"""
Abstrakcyjna klasa bazowa backendów embeddingowych.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingBackend(ABC):
    """Interfejs, który musi spełniać każdy backend."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Zwraca listę wektorów o długości zależnej od modelu.

        Args:
            texts: kolekcja dowolnych ciągów znaków
        """
        raise NotImplementedError
