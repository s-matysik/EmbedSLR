"""
embeddings/base.py
──────────────────
Abstrakcyjna (interfejsowa) klasa bazowa dla wszystkich backendów
generujących osadzenia tek­stu.  Plik zawiera również alias
`EmbeddingBackend` – utrzymy­wany dla wstecznej zgodności z
importami w `embedslr/__init__.py` (i ew. starym kodem użytkowników).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence


class BaseEmbedder(ABC):
    """Bazowa klasa dla każdego back‑endu embeddingów."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Zwróć listę wektorów osadzeń odpowiadających podanej sekwencji tekstów.

        Parameters
        ----------
        texts : Sequence[str]
            Kolekcja (iterowalna) tekstów do zosadzenia.

        Returns
        -------
        List[List[float]]
            Lista wektorów (lista floatów) – jeden wektor na każdy tekst.
        """
        raise NotImplementedError

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        """Pozwala wywołać instancję jak funkcję (`embeddings = backend(texts)`)."""
        return self.embed(texts)


# ─── alias wstecznej zgodności ────────────────────────────────────────────────
# Dzięki temu stary import:
#     from embeddings.base import EmbeddingBackend
# nadal będzie działał.
EmbeddingBackend = BaseEmbedder
