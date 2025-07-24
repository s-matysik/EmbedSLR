"""
EmbedSLR – Single‑Line‑Reference Embeddings
https://github.com/rafalposwiata/EmbedSLR
"""

from importlib import metadata
import importlib
import sys as _sys

# ──────────────────────────────────────────────────────────
# 1)  Ładujemy *właściwy* podpakiet ``embeddings``.
#     Po reorganizacji repozytorium może on leżeć w dwóch miejscach:
#       • embedslr/embeddings      (nowy układ, zalecany)
#       • top‑level  embeddings/   (stary układ)
try:
    from . import embeddings as _emb_subpkg               # noqa: F401
except ModuleNotFoundError:
    _emb_subpkg = importlib.import_module("embeddings")

# Zarejestruj aliasy, żeby każdy wariant importu działał:
#   import embedslr.embeddings         ✔
#   import embeddings.base             ✔
_sys.modules.setdefault(f"{__name__}.embeddings", _emb_subpkg)
_sys.modules.setdefault("embeddings", _emb_subpkg)

# ──────────────────────────────────────────────────────────
# 2)  Analogiczne aliasy dla pozostałych pakietów toplevel:
for _alias in ("utils", "ranking", "biblio", "io"):
    try:
        _pkg = importlib.import_module(_alias)
        _sys.modules.setdefault(f"{__name__}.{_alias}", _pkg)
    except ModuleNotFoundError:
        # brak danego podpakietu – ignorujemy
        pass

# ──────────────────────────────────────────────────────────
# 3)  Metadane
try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["embeddings", "__version__"]
