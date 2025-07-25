"""
EmbedSLR – Embedding‑based Screening for Systematic Literature Reviews
https://github.com/s-matysik/EmbedSLR
"""
from __future__ import annotations

import importlib
import sys as _sys
from importlib import metadata

# ────────────────────────────────────────────────────────
# 1)  Załaduj podpakiet „embeddings” *po* zakończeniu init.
def _import_subpkg(name: str):
    """
    Importuje podpakiet najpierw jako „embedslr.<name>”, a gdy to się nie uda,
    próbuje wariantu legacy – top‑level „<name>”.
    Zwraca zaimportowany moduł.
    """
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError:
        return importlib.import_module(name)


# embeddings (główna funkcjonalność pakietu)
_emb_subpkg = _import_subpkg("embeddings")

# rejestrujemy aliasy w sys.modules
_sys.modules.setdefault(f"{__name__}.embeddings", _emb_subpkg)
_sys.modules.setdefault("embeddings", _emb_subpkg)

# ────────────────────────────────────────────────────────
# 2)  Aliasujemy pozostałe stare pakiety top‑level, jeśli istnieją
for _alias in ("utils", "ranking", "biblio", "io"):
    try:
        _pkg = importlib.import_module(_alias)
    except ModuleNotFoundError:
        continue
    _sys.modules.setdefault(f"{__name__}.{_alias}", _pkg)

# ────────────────────────────────────────────────────────
# 3)  Metadane
try:
    __version__ = metadata.version("embedslr")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["embeddings", "__version__"]
