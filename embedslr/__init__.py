"""
EmbedSLR
========
Główny pakiet wysokopoziomowych interfejsów do:
 • pobierania i łącznia danych bibliograficznych,
 • generowania embeddingów (wrappers: SBERT, OpenAI, Cohere, Nomic, Jina),
 • rankingowania publikacji względem zapytania badawczego,
 • obliczeń bibliometrycznych.

API na razie w fazie *beta* – może się zmieniać bez zapowiedzi.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("embedslr")
except PackageNotFoundError:   # pakiet instalowany w editable ‑e
    __version__ = "0.0.0+editable"

# ─── public shortcuts ────────────────────────────────────────────────────────
from embeddings.base import EmbeddingBackend          # noqa: E402 F401
from ranking.ranker import Ranker                     # noqa: E402 F401
from biblio.metrics import BibliometricAnalyzer       # noqa: E402 F401
