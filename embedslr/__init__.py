from importlib import metadata as _meta
from .embeddings import get_embeddings, list_models
from .similarity import rank_by_cosine
from .bibliometrics import full_report, indicators
from .colab_app import run as colab_run

try:
    __version__ = _meta.version(__name__)
except _meta.PackageNotFoundError:
    __version__ = "0.4.0"

__all__ = [
    "get_embeddings",
    "list_models",
    "rank_by_cosine",
    "full_report",
    "indicators",
    "colab_run",
]
