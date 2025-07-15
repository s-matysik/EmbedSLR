import re
import pandas as pd
from typing import Set, Tuple

def parse_author_fragment(fr: str) -> Tuple[str, str]:
    fr = re.sub(r"[.,]+$", "", fr.strip()).lower()
    parts = re.split(r"\s+", fr)
    if not parts:
        return ("", "")
    last_name = parts[0]
    initial   = parts[1][0] if len(parts) > 1 and len(parts[1]) == 1 else ""
    return last_name, initial

def parse_authors_column(col: pd.Series) -> list[Set[Tuple[str, str]]]:
    return [ {parse_author_fragment(a) for a in (row or "").split(";") if a.strip()}
             for row in col ]
