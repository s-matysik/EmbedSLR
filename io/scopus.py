from pathlib import Path
import pandas as pd

POSSIBLE_TITLE_COLS  = ['Article Title', 'Title', 'TI']
POSSIBLE_ABS_COLS    = ['Abstract', 'AB']

def _find_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None

def load_scopus_csv(path: str | Path) -> pd.DataFrame:
    """
    Ładuje plik CSV wyeksportowany ze Scopusa i dodaje kolumnę `combined_text`.
    """
    df = pd.read_csv(path, sep=",", low_memory=False)
    title_col    = _find_first(df, POSSIBLE_TITLE_COLS)
    abstract_col = _find_first(df, POSSIBLE_ABS_COLS)

    df["combined_text"] = (
        df[title_col].fillna("").astype(str) + " " +
        df[abstract_col].fillna("").astype(str)
    )
    return df
