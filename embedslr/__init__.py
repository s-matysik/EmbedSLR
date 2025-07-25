"""
EmbedSLR – biblioteka do systematycznych przeglądów literatury oparta na
embeddingach.

Kluczowe klasy eksportowane publicznie:
    • EmbeddingBackend      – abstrakcyjny interfejs backendów
    • Ranker                – ranking rekordów na podstawie embedów
    • BibliometricAnalyzer  – proste metryki bibliometryczne
"""

from importlib.metadata import PackageNotFoundError, version

try:                     # ← wersja zainstalowana z PyPI / editable
    __version__: str = version(__name__)
except PackageNotFoundError:      # ← brak metadanych (np. zip)
    __version__ = "0.0.dev0"

# ── PUBLICZNE SKRÓTY (wyłącznie importy względne!) ─────────────────
from .embeddings.base import EmbeddingBackend        # noqa: F401,E402
from .ranking.ranker import Ranker                   # noqa: F401,E402
from .biblio.metrics import BibliometricAnalyzer     # noqa: F401,E402

__all__ = [
    "EmbeddingBackend",
    "Ranker",
    "BibliometricAnalyzer",
]
