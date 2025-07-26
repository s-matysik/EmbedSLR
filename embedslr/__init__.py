from importlib import metadata as _meta
from .embeddings import get_embeddings, list_models
from .similarity import rank_by_cosine
from .bibliometrics import full_report
from .colab_app import run as colab_run

try:
    __version__ = _meta.version(__name__)
except _meta.PackageNotFoundError:
    from ._version import __version__  # noqa: F401

__all__ = [
    "get_embeddings",
    "list_models",
    "rank_by_cosine",
    "full_report",
    "colab_run",
]
