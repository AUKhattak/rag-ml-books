import hashlib
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CacheManager:
    """Simple cache for embeddings to avoid recomputing"""

    def __init__(self, cache_dir: str = "data/cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.cache = self._load_cache()

        # Dirty flag: avoid saving on every set()
        self._dirty = False

    def _load_cache(self) -> dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
            self._dirty = False
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _generate_key(self, chunk_id: str) -> str:
        """Generate cache key"""
        return hashlib.md5(chunk_id.encode()).hexdigest()

    def get(self, chunk_id: str) -> list[float] | None:
        """Get cached embedding if exists"""
        key = self._generate_key(chunk_id)
        return self.cache.get(key)

    def set(self, chunk_id: str, embedding: list[float]):
        """
        Cache embedding (in-memory only).
        Call save() to persist to disk.
        """
        key = self._generate_key(chunk_id)
        self.cache[key] = embedding
        self._dirty = True

    def save(self):
        """Persist cache to disk (only if dirty)"""
        if self._dirty:
            self._save_cache()
            logger.info(f"💾 Cache saved: {len(self.cache)} entries")

    def get_batch(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """Get multiple cached embeddings"""
        results = {}
        for chunk_id in chunk_ids:
            embedding = self.get(chunk_id)
            if embedding is not None:
                results[chunk_id] = embedding
        return results

    def clear(self):
        """Clear all cached embeddings"""
        self.cache = {}
        self._save_cache()
        logger.info("🗑️ Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        if not self.cache:
            return {"total_embeddings": 0, "cache_size_mb": 0.0, "model_dimension": 0}

        # Estimate size
        size_bytes = sys.getsizeof(self.cache)
        for k, v in self.cache.items():
            size_bytes += sys.getsizeof(k)
            if v:
                size_bytes += len(v) * 8  # 8 bytes per float64

        return {
            "total_embeddings": len(self.cache),
            "cache_size_mb": size_bytes / (1024 * 1024),
            "model_dimension": len(next(iter(self.cache.values()))) if self.cache else 0,
        }
