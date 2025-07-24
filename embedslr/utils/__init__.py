"""
Alias‑package, który pozwala zachować dotychczasowe importy:

    from embedslr.utils.batching import chunk_list

Faktyczny kod znajduje się w top‑level pakiecie ``utils``.
"""
import importlib
import sys
from types import ModuleType

# 1) załaduj prawdziwy moduł utils.*
_real_utils: ModuleType = importlib.import_module("utils")

# 2) zarejestruj alias – `import embedslr.utils` →  otrzymuje ten sam obiekt
sys.modules.setdefault(__name__, _real_utils)

# 3) alias dla embedslr.utils.batching
_real_batching = importlib.import_module("utils.batching")
sys.modules.setdefault(f"{__name__}.batching", _real_batching)

# eksportujemy przynajmniej chunk_list
from utils.batching import chunk_list   # noqa: F401,E402
__all__ = ["chunk_list"]
