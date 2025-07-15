import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

def rank_dataframe(df: pd.DataFrame,
                   query: str,
                   embedder,
                   text_col: str = "combined_text") -> pd.DataFrame:
    """
    Zwraca DataFrame posortowany wg rosnącej odległości kosinusowej
    pomiędzy embeddingiem `query` a embeddingami kolumny `text_col`.
    """
    # 1) Embedding query
    query_vec: List[float] = embedder.encode([query])[0]

    # 2) Embedding artykułów
    art_vecs: List[List[float]] = embedder.encode(df[text_col].tolist())

    # 3) Cosine similarity
    sim = cosine_similarity([np.array(query_vec)], np.array(art_vecs))[0]
    df_out = df.copy()
    df_out["distance_cosine"] = 1 - sim
    return df_out.sort_values("distance_cosine", ascending=True)
