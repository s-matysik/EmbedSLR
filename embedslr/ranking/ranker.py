"""
Najprostsza możliwa implementacja rankera ‑ tylko żeby zaspokoić importy.
"""

from __future__ import annotations
import numpy as np


class Ranker:                                    # noqa: D101
    def __init__(self, metric: str = "cosine") -> None:
        if metric not in {"cosine", "dot"}:
            raise ValueError("metric must be 'cosine' or 'dot'")
        self.metric = metric

    # --------------------------------------------------------------------- #
    # API‐użytkownika – przykład minimum, wystarczy na początek            #
    # --------------------------------------------------------------------- #
    def score(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """
        Zwraca wektor podobieństw query ‑–> wszystkie dokumenty.
        """
        query_emb = query_emb / np.linalg.norm(query_emb)
        doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
        if self.metric == "cosine":
            return doc_embs @ query_emb
        return doc_embs @ query_emb  # dot product identyczny po normalizacji
