
"""
Pod‑pakiet z konkretnymi backendami embeddingów.

Obecnie dostępne:
    • OpenAIEmbedder  – API OpenAI / Azure OpenAI
"""

from .openai_api import OpenAIEmbedder  # noqa: F401

__all__ = ["OpenAIEmbedder"]
