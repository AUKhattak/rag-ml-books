"""
Embedder - generates embeddings for text chunks with caching and batching.

Usage:
    embedder = Embedder(model_name='bge-small', batch_size=32, use_cache=True)
    embeddings = embedder.embed_chunks(chunks)
    embedder.save_embeddings(embeddings, chunks, "data/embeddings/v1.parquet")
"""

import json
import logging
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from catalog.embedding_models import ModelConfig

from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class Embedder:
    """Main embedding generator with built-in batching and caching"""

    def __init__(self, model_name: str = "bge-small", batch_size: int = 32, use_cache: bool = True):
        """
        Initialize the embedder.

        Args:
            model_name: Short key from ModelConfig (e.g., 'bge-small', 'bge-large')
            batch_size: Number of chunks to encode per batch
            use_cache: Whether to cache embeddings
        """
        # Resolve short key → real HuggingFace name
        self.config = ModelConfig.get_config(model_name)
        self.model_key = model_name
        self.model_name = self.config["name"]  # 'BAAI/bge-small-en-v1.5'
        self.model_dimension = self.config["dimension"]
        self.batch_size = batch_size or self.config.get("batch_size", 32)
        self.use_cache = use_cache

        # Load model
        logger.info(f"📦 Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"✅ Model loaded: {self.model_dimension} dimensions")

        # Cache
        self.cache = CacheManager() if use_cache else None

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        """
        Embed chunks with caching + batching.

        Args:
            chunks: List of dicts with at least 'id' and 'content' keys

        Returns:
            List of embedding vectors (one per chunk, in same order)
        """
        if not chunks:
            return []

        texts = [chunk.get("content", "") for chunk in chunks]
        chunk_ids = [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)]

        # Pre-allocate results (preserves order)
        embeddings: list[list[float] | None] = [None] * len(chunks)

        # ─── Step 1: Check cache ───
        to_compute_indices = []
        to_compute_texts = []
        cache_hits = 0

        for i, chunk_id in enumerate(chunk_ids):
            if self.cache:
                cached = self.cache.get(chunk_id)
                if cached is not None:
                    embeddings[i] = cached
                    cache_hits += 1
                    continue

            to_compute_indices.append(i)
            to_compute_texts.append(texts[i])

        logger.info(
            f"📊 Cache: {cache_hits} hits, {len(to_compute_texts)} misses (total {len(chunks)})"
        )

        # ─── Step 2: Batch-encode cache misses ───
        if to_compute_texts:
            num_batches = (len(to_compute_texts) + self.batch_size - 1) // self.batch_size

            for batch_num, start in enumerate(
                range(0, len(to_compute_texts), self.batch_size), start=1
            ):
                batch_texts = to_compute_texts[start : start + self.batch_size]
                batch_indices = to_compute_indices[start : start + self.batch_size]

                batch_embeddings = self.model.encode(
                    batch_texts, show_progress_bar=False, convert_to_numpy=True
                )

                for idx, emb in zip(batch_indices, batch_embeddings, strict=True):
                    emb_list = emb.tolist()
                    embeddings[idx] = emb_list

                    if self.cache:
                        self.cache.set(chunk_ids[idx], emb_list)

                logger.info(f"   Batch {batch_num}/{num_batches} done")

        # ─── Step 3: Flush cache to disk ───
        if self.cache:
            self.cache.save()

        return embeddings  # type: ignore

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model_dimension

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {
            "total_embeddings": 0,
            "cache_size_mb": 0.0,
            "model_dimension": self.model_dimension,
        }

    def save_embeddings(self, embeddings: list[list[float]], chunks: list[dict], output_file: str):
        """
        Save embeddings + metadata to disk.

        Args:
            embeddings: List of embedding vectors
            chunks: List of original chunk dicts (must align with embeddings)
            output_file: Path ending in .parquet or .json
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for emb, chunk in zip(embeddings, chunks, strict=True):
            data.append(
                {
                    "chunk_id": chunk.get("id", ""),
                    "book_title": chunk.get("book_title", ""),
                    "page_number": chunk.get("page_number", 0),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "content": chunk.get("content", ""),
                    "embedding": emb,
                    "embedding_dim": len(emb) if emb else 0,
                }
            )

        if output_file.endswith(".parquet"):
            import pandas as pd

            pd.DataFrame(data).to_parquet(output_file, index=False)
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

        logger.info(f"💾 Saved {len(embeddings)} embeddings to {output_file}")
