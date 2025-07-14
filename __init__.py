"""
embedslr
~~~~~~~~
Biblioteka do wspomagania systematycznych przeglądów literatury
z wykorzystaniem embeddingów i podstawowych metryk bibliometrycznych.

Dokumentacja: https://github.com/<ORG>/embedslr
Licencja: MIT
"""
from importlib import metadata as _md
__version__: str = _md.version(__name__) if _md else "0.0.0"  # pragma: no cover
