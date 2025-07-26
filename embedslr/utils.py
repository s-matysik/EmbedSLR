from __future__ import annotations
import os
from contextlib import contextmanager
from tqdm import tqdm


def chunk_iterable(seq, n: int = 96):
    """Yield successive *n*-sized chunks from *seq*."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def getenv_or_raise(var: str, provider: str):
    val = os.getenv(var)
    if not val:
        raise RuntimeError(
            f"{provider} API key not found – set environment variable {var}=<key>."
        )
    return val


@contextmanager
def progress(desc: str, total: int | None = None):
    bar = tqdm(total=total, desc=desc, unit="it")
    try:
        yield bar
    finally:
        bar.close()
