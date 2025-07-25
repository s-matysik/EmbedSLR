"""
EmbedSLR
========
Biblioteka do tworzenia i oceny rankingów na podstawie embeddingów artykułów
naukowych (S/L/R = Search/Link/Recommend).

Wersja minimalna – zawiera jedynie skróty importowe, żeby użytkownik mógł
zrobić ::

    import embedslr
    ranker = embedslr.Ranker(...)
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("embedslr")
except PackageNotFoundError:  # instalacja editable → brak metadanych
    __version__ = "0.0.0.dev0"

# ── PUBLICZNE SKRÓTY (wyłącznie importy względne!) ──────────────────────
from .embeddings.base import EmbeddingBackend        # noqa: F401,E402
from .ranking.ranker import Ranker                   # noqa: F401,E402
from .biblio.metrics import BibliometricAnalyzer     # noqa: F401,E402

__all__ = [
    "EmbeddingBackend",
    "Ranker",
    "BibliometricAnalyzer",
]
