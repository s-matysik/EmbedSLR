"""
embedslr – pakiet główny
────────────────────────
Ten moduł:
 • udostępnia przyjazne aliasy klas najwyższego poziomu,
 • konfiguruje namespace „embedslr.utils.*”, aby zachować kompatybilność
   ze starszym kodem, w którym występowały importy typu
   `from embedslr.utils.batching import chunk_list`.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

__all__: list[str] = [
    "EmbeddingBackend",
    "Ranker",
    "BibliometricAnalyzer",
    "__version__",
]

__version__ = "0.1.0"

# ─── publiczne skróty (re‑export) ─────────────────────────────────────────────
from embeddings.base import BaseEmbedder as EmbeddingBackend  # noqa: E402
from ranking.ranker import Ranker  # noqa: E402
from biblio.metrics import BibliometricAnalyzer  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# ★ Sekcja kompatybilności:
#   mapujemy istniejący, płaski pakiet „utils” na namespace „embedslr.utils”
#   tak, aby wszystkie importy w stylu  embedslr.utils.*  nadal działały,
#   mimo że realny kod znajduje się w katalogu top‑level „utils/…”.
# ──────────────────────────────────────────────────────────────────────────────
try:
    # Załaduj pakiet „utils” (musi istnieć, bo jest instalowany).
    _utils_pkg: ModuleType = importlib.import_module("utils")

    # Zarejestruj go pod dodatkową ścieżką importu „embedslr.utils”.
    sys.modules.setdefault("embedslr.utils", _utils_pkg)

    # Zarejestruj podścieżkę  embedslr.utils.batching
    #  => wskazuje dokładnie na utils.batching
    _batching_mod: ModuleType = importlib.import_module("utils.batching")
    sys.modules.setdefault("embedslr.utils.batching", _batching_mod)

except ModuleNotFoundError:  # pragma: no cover
    # Teoretycznie nie powinno się zdarzyć – „utils” jest częścią projektu.
    pass
