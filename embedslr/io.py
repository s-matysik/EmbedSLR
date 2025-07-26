from __future__ import annotations
import pandas as pd

# heuristic column names incl. Scopus export
POSSIBLE_TITLE = ["Article Title", "Title", "TI", "Document Title"]
POSSIBLE_ABS = ["Abstract", "AB", "Description"]


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _first(candidates: list[str], df: pd.DataFrame) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No column among {candidates} found.")


def autodetect_columns(df: pd.DataFrame) -> tuple[str, str]:
    return _first(POSSIBLE_TITLE, df), _first(POSSIBLE_ABS, df)


def combine_title_abstract(
    df: pd.DataFrame, title_col: str, abs_col: str
) -> pd.Series:
    return df[title_col].fillna("") + " " + df[abs_col].fillna("")
