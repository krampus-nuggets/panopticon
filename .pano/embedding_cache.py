"""Persistent KV cache for embedding vectors, backed by shelve."""

import hashlib
import shelve


class EmbeddingCache:
    """Maps chunk text to precomputed embedding vectors."""

    def __init__(self, cache_path: str, model_name: str):
        self.model_name = model_name
        self.store = shelve.open(cache_path)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        return self.store.get(self._key(text))

    def put(self, text: str, vector: list[float]) -> None:
        self.store[self._key(text)] = vector

    def close(self) -> None:
        self.store.close()
