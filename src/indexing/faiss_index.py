"""
FAISS-based vector index for the RAG pipeline.

Wraps FAISS's IndexFlatIP so that vectors are searchable by cosine similarity,
with a parallel metadata store so that search results can be traced back to
their source chunks.

Design notes:
    - Embeddings are normalized to unit length at add time and query time.
      With unit-length vectors, inner product == cosine similarity.
    - Metadata is kept out of the FAISS index and stored as a Parquet file
      with the same row order. Row index i in FAISS corresponds to row i in
      the metadata file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FaissIndex:
    """Vector index built on FAISS's IndexFlatIP."""

    def __init__(self, dimension: int):
        """
        Initialize an empty FAISS index.

        Args:
            dimension: Dimensionality of the vectors to be indexed (e.g. 384 for bge-small).
        """
        self.dimension = dimension
        self.index: faiss.Index | None = None
        self.metadata: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        normalize: bool = True,
    ) -> None:
        """
        Construct the FAISS index from a matrix of embeddings.

        Args:
            embeddings: (N, D) array of float32 vectors.
            metadata:   DataFrame with N rows. Row i corresponds to embeddings[i].
            normalize:  If True, L2-normalize the embeddings (cosine similarity).
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != declared dimension {self.dimension}"
            )
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"Metadata rows ({len(metadata)}) != embeddings ({embeddings.shape[0]})"
            )

        vectors = embeddings.astype("float32")

        if normalize:
            vectors = self._normalize(vectors)

        # IndexFlatIP: exact inner product search (== cosine on normalized vectors).
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)  # type: ignore[arg-type]

        # Keep metadata in memory too (handy for debugging).
        self.metadata = metadata.reset_index(drop=True)

        logger.info(
            f"Built FAISS index: {self.index.ntotal} vectors, "
            f"dimension {self.dimension}, normalize={normalize}"
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """
        Save the FAISS index and its parallel metadata to disk.

        Args:
            index_path:    Path to write the .faiss binary file.
            metadata_path: Path to write the .parquet metadata file.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Cannot save: index has not been built.")

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        self.metadata.to_parquet(metadata_path, index=False)

        logger.info(f"Saved FAISS index  → {index_path}")
        logger.info(f"Saved metadata      → {metadata_path}")

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> FaissIndex:
        """
        Load a FAISS index and its metadata from disk.

        Args:
            index_path:    Path to the .faiss binary file.
            metadata_path: Path to the .parquet metadata file.

        Returns:
            A FaissIndex instance ready for search.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        faiss_index = faiss.read_index(str(index_path))
        metadata = pd.read_parquet(metadata_path)

        obj = cls(dimension=faiss_index.d)
        obj.index = faiss_index
        obj.metadata = metadata

        logger.info(
            f"Loaded FAISS index: {faiss_index.ntotal} vectors, "
            f"dimension {faiss_index.d}, metadata rows {len(metadata)}"
        )
        return obj

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        normalize: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search the index for the k nearest vectors.

        Args:
            query:     1D (D,) or 2D (1, D) array with the query vector.
            k:         Number of results to return.
            normalize: If True, L2-normalize the query (matches build-time normalization).

        Returns:
            A list of dicts, each containing:
                - score:      cosine similarity (higher = more similar)
                - row:        row index in the metadata DataFrame
                - metadata:   the metadata row as a dict
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Cannot search: index has not been built or loaded.")

        q = np.asarray(query, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(f"Query dim {q.shape[1]} != index dimension {self.dimension}")

        if normalize:
            q = self._normalize(q)

        # distances: (1, k) inner products; indices: (1, k) row positions
        distances, indices = self.index.search(q, k)  # type: ignore[arg-type]

        results: list[dict[str, Any]] = []
        for score, row in zip(distances[0], indices[0], strict=True):
            if row == -1:  # FAISS pads with -1 if fewer than k results
                continue
            results.append(
                {
                    "score": float(score),
                    "row": int(row),
                    "metadata": self.metadata.iloc[int(row)].to_dict(),
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Return a summary of the index."""
        if self.index is None:
            return {"built": False}
        return {
            "built": True,
            "ntotal": int(self.index.ntotal),
            "dimension": int(self.index.d),
            "metadata_rows": len(self.metadata) if self.metadata is not None else 0,
            "index_type": type(self.index).__name__,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize each row to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Guard against division by zero (should not happen for real embeddings).
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
