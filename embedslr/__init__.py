"""
EmbedSLR
========
Wysokopoziomowe API do:
• pobierania i łączenia danych bibliograficznych,
• generowania embeddingów (SBERT, OpenAI, Cohere, Nomic, Jina),
• rankingowania publikacji względem zapytania badawczego,
• obliczeń bibliometrycznych.

Stan: wczesna wersja 0.1 – interfejs może jeszcze ulec zmianie.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("embedslr")
except PackageNotFoundError:          # pakiet instalowany w trybie ‑e
    __version__ = "0.0.0+editable"

# ─── public shortcuts ────────────────────────────────────────────────────────
from embeddings.base import EmbeddingBackend           # noqa: F401,E402
from ranking.ranker import Ranker                      # noqa: F401,E402
from biblio.metrics import BibliometricAnalyzer        # noqa: F401,E402

__all__ = ["EmbeddingBackend", "Ranker",
           "BibliometricAnalyzer", "__version__"]
