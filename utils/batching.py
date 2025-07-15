# embedslr/utils/batching.py
from typing import Iterable, Generator, TypeVar

T = TypeVar("T")


def chunk_list(data: Iterable[T], chunk_size: int = 100) -> Generator[list[T], None, None]:
    """Dzieli iterowalny obiekt `data` na podlisty po `chunk_size` elementów."""
    chunk: list[T] = []
    for item in data:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
