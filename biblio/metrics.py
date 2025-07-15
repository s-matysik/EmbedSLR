from __future__ import annotations
import pandas as pd
from itertools import combinations
from .parser import parse_authors_column

class CorpusMetrics:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def jaccard_keywords(self) -> float:
        kws = self.df["Author Keywords"].apply(
            lambda x: set(k.strip().lower() for k in str(x).split(";") if k.strip())
        )
        num, den = 0.0, 0
        for a, b in combinations(kws, 2):
            inter = len(a & b)
            union = len(a | b)
            if union:
                num += inter / union
                den += 1
        return num / den if den else 0.0
