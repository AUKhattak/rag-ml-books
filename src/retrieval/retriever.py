"""
Retriever: query → embed → search FAISS → ranked chunks.

Loads a FAISS index and a SentenceTransformer once at construction time.
The model key must match the one used to embed the documents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from catalog.embedding_models import ModelConfig
from indexing.faiss_index import FaissIndex

logger = logging.getLogger(__name__)


class Retriever:
    """Query the FAISS index for the top-k most similar chunks."""

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        model_key: str = "bge-small",
    ):
        """
        Load the FAISS index, its metadata, and the embedding model.

        Args:
            index_path:    Path to the .faiss index file.
            metadata_path: Path to the aligned .parquet metadata file.
            model_key:     Catalog key (e.g. 'bge-small'). Resolved via ModelConfig.
        """
        cfg = ModelConfig.get_config(model_key)
        self.model_key = model_key
        self.model_name = cfg["name"]
        self.model_dimension = cfg["dimension"]

        logger.info(f"Loading index from {index_path}")
        self.index = FaissIndex.load(index_path, metadata_path)

        # Verify the index was built with the same model.
        if "model_key" in self.index.metadata.columns:
            stored_key = self.index.metadata["model_key"].iloc[0]
            if stored_key != self.model_key:
                raise ValueError(
                    f"Model mismatch: index was built with '{stored_key}', "
                    f"but Retriever is configured with '{self.model_key}'."
                )
            logger.info(f"✅ Model key verified: {self.model_key}")

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        stats = self.index.stats()
        logger.info(
            f"Retriever ready: {stats['ntotal']} vectors, dimension {stats['dimension']}"
        )

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Return the k most similar chunks for a query string.
        """
        if not query.strip():
            return []

        query_vec = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        raw_results = self.index.search(
            np.asarray(query_vec, dtype="float32"),
            k=k,
            normalize=True,
        )

        results: list[dict[str, Any]] = []
        for r in raw_results:
            meta = r["metadata"]
            results.append(
                {
                    "score": r["score"],
                    "row": r["row"],
                    "chunk_id": meta.get("chunk_id", ""),
                    "book": meta.get("book_title", ""),
                    "page": meta.get("page_number", 0),
                    "content": meta.get("content", ""),
                }
            )
        return results

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        return self.index.stats()