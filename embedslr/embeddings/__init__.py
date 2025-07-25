"""
Podpakiet z backendami embeddingów.

Użytkownik może::

    from embedslr.embeddings.openai_api import OpenAIEmbedder
"""
from .base import EmbeddingBackend        # noqa: F401
from .openai_api import OpenAIEmbedder    # noqa: F401

__all__ = ["EmbeddingBackend", "OpenAIEmbedder"]
