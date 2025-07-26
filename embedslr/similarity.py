from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def rank_by_cosine(
    query_vec: list[float], doc_vecs: list[list[float]], df: pd.DataFrame
) -> pd.DataFrame:
    q = np.asarray(query_vec).reshape(1, -1)
    d = np.asarray(doc_vecs)
    sim = cosine_similarity(q, d)[0]
    out = df.copy()
    out["distance_cosine"] = 1 - sim
    return out.sort_values("distance_cosine")
