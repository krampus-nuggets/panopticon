from pathlib import Path

import pytest

from embedding_cache import EmbeddingCache


class TestEmbeddingCacheGetPut:
    """Basic get/put operations."""

    def test_put_and_get(self, tmp_path: Path):
        cache = EmbeddingCache(str(tmp_path / "cache"), "model-a")
        try:
            cache.put("hello world", [1.0, 2.0, 3.0])
            result = cache.get("hello world")
            assert result == [1.0, 2.0, 3.0]
        finally:
            cache.close()

    def test_get_miss_returns_none(self, tmp_path: Path):
        cache = EmbeddingCache(str(tmp_path / "cache"), "model-a")
        try:
            assert cache.get("not stored") is None
        finally:
            cache.close()

    def test_overwrite_existing_key(self, tmp_path: Path):
        cache = EmbeddingCache(str(tmp_path / "cache"), "model-a")
        try:
            cache.put("text", [1.0])
            cache.put("text", [2.0])
            assert cache.get("text") == [2.0]
        finally:
            cache.close()


class TestModelScopedKeys:
    """Keys are scoped by model name to prevent cross-model collisions."""

    def test_different_models_different_keys(self, tmp_path: Path):
        cache_a = EmbeddingCache(str(tmp_path / "cache"), "model-a")
        cache_b = EmbeddingCache(str(tmp_path / "cache"), "model-b")
        try:
            cache_a.put("same text", [1.0])
            assert cache_b.get("same text") is None
        finally:
            cache_a.close()
            cache_b.close()

    def test_same_model_shares_keys(self, tmp_path: Path):
        cache1 = EmbeddingCache(str(tmp_path / "cache"), "model-a")
        try:
            cache1.put("text", [1.0, 2.0])
        finally:
            cache1.close()

        cache2 = EmbeddingCache(str(tmp_path / "cache"), "model-a")
        try:
            assert cache2.get("text") == [1.0, 2.0]
        finally:
            cache2.close()


class TestPersistence:
    """Cache persists across close/reopen cycles."""

    def test_data_survives_close(self, tmp_path: Path):
        cache = EmbeddingCache(str(tmp_path / "cache"), "model-a")
        cache.put("key", [0.5, 0.5])
        cache.close()

        cache2 = EmbeddingCache(str(tmp_path / "cache"), "model-a")
        try:
            assert cache2.get("key") == [0.5, 0.5]
        finally:
            cache2.close()
