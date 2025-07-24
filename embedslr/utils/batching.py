"""
Thin shim: re‑eksport funkcji z top‑level utils.batching.
Pozwala, by istniał fizyczny plik (nie tylko alias w sys.modules).
"""
from utils.batching import *            # noqa: F403,F401
