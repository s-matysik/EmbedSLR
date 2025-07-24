"""
EmbedSLR – Single‑Line‑Reference Embeddings
https://github.com/rafalposwiata/EmbedSLR
"""

from importlib import metadata
import sys as _sys

# Upewniamy się, że **wewnętrzny** pakiet embedslr.embeddings
# jest również dostępny pod nazwą 'embeddings'.
# Dzięki temu absolutny import `import embeddings.base`
# zawsze wskazuje na właściwy moduł, nawet gdy w systemie jest
# obca paczka o tej samej nazwie.
from . import embeddings as _emb_subpkg
_sys.modules.setdefault("embeddings", _emb_subpkg)

# ─── publiczne skróty ────────────────────────────────────────────────────────
from .embeddings.base import EmbeddingBackend            # noqa: F401,E402
from .ranking.ranker import Ranker                       # noqa: F401,E402
from .biblio.metrics import BibliometricAnalyzer         # noqa: F401,E402

__all__ = [
    "EmbeddingBackend",
    "Ranker",
    "BibliometricAnalyzer",
]

__version__ = metadata.version("embedslr")
