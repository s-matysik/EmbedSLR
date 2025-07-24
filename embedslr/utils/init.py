"""
Alias‑package pozwalający zachować stare importy:

    from embedslr.utils.batching import chunk_list

Faktyczny kod znajduje się w top‑level package `utils`.
"""
import importlib
import sys
from types import ModuleType

# ────────────────────────────────────────────────────────────────────────────
# 1) załaduj prawdziwy moduł utils.*
_real_utils: ModuleType = importlib.import_module("utils")

# 2) zarejestruj alias w sys.modules  →  `import embedslr.utils` zadziała
sys.modules.setdefault(__name__, _real_utils)

# 3) upewnij się, że pod‑moduł batching także ma alias
_real_batching = importlib.import_module("utils.batching")
sys.modules.setdefault(f"{__name__}.batching", _real_batching)

# co najmniej chunk_list eksportujemy jawnie
from utils.batching import chunk_list   # noqa: F401,E402
__all__ = ["chunk_list"]
