"""
EmbedSLR
========
Biblioteka do tworzenia i oceny rankingów artykułów naukowych
z wykorzystaniem embeddingów (S/L/R = Search / Link / Recommend).

Użytkownik może napisać::

    import embedslr
    ranker = embedslr.Ranker(...)
"""

from importlib import metadata as _metadata

# ────────────────────────────────────────────────────────────────────────
#  Wersja pakietu
# ────────────────────────────────────────────────────────────────────────
try:
    __version__ = _metadata.version(__name__)          # wersja z metadanych pakietu
except _metadata.PackageNotFoundError:                 # tryb editable / brak instalacji
    __version__ = "0.0.0.dev0"

# ────────────────────────────────────────────────────────────────────────
#  Importy wysokiego poziomu (API publiczne)
# ────────────────────────────────────────────────────────────────────────
from .embeddings.base import EmbeddingBackend          # noqa: E402,F401
from .embeddings.openai_api import OpenAIEmbedder      # noqa: E402,F401
from .ranking.ranker import Ranker                     # noqa: E402,F401
from .biblio.metrics import BibliometricAnalyzer       # noqa: E402,F401

__all__: list[str] = [
    "EmbeddingBackend",
    "OpenAIEmbedder",
    "Ranker",
    "BibliometricAnalyzer",
    "__version__",
]
