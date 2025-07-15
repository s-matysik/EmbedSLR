from typing import List
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder

class LocalSBERT(BaseEmbedder):
    """Embeddingi generowane lokalnym modelem Sentence‑Transformers."""

    def __init__(self,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 batch_size: int = 96) -> None:
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts,
                                 batch_size=self.batch_size,
                                 show_progress_bar=False).tolist()
