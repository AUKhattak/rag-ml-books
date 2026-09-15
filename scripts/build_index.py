"""
Build a FAISS index from the saved embeddings.

Usage:
    python scripts/build_index.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from indexing.faiss_index import FaissIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────── #
EMBEDDINGS_FILE = "data/processed/embeddings/embeddings_v1.parquet"
INDEX_FILE = "data/processed/indexes/books_v1.faiss"
METADATA_FILE = "data/processed/indexes/books_v1_meta.parquet"

EXPECTED_DIM = 384  # bge-small-en-v1.5 output dimension

# Columns to strip from the embeddings DataFrame before saving as metadata.
# The 'embedding' column is dropped because it lives in the FAISS index.
EMBEDDING_COLUMN = "embedding"


def main() -> None:
    logger.info("=" * 60)
    logger.info("📇 BUILDING FAISS INDEX")
    logger.info("=" * 60)

    # ─── 1. Load embeddings ──────────────────────────────────────── #
    emb_path = Path(EMBEDDINGS_FILE)
    if not emb_path.exists():
        logger.error(f"❌ Embeddings file not found: {EMBEDDINGS_FILE}")
        logger.info("💡 Run generate_embeddings.py first.")
        sys.exit(1)

    logger.info(f"📄 Loading embeddings: {EMBEDDINGS_FILE}")
    df = pd.read_parquet(emb_path)
    logger.info(f"✅ Loaded {len(df)} rows")

    if EMBEDDING_COLUMN not in df.columns:
        logger.error(f"❌ Column '{EMBEDDING_COLUMN}' not found in embeddings file.")
        logger.info(f"   Available columns: {list(df.columns)}")
        sys.exit(1)

    # ─── 2. Extract vectors as (N, D) float32 array ──────────────── #
    logger.info("🔢 Stacking vectors into a matrix...")
    vectors = np.stack(df[EMBEDDING_COLUMN].values).astype("float32")
    logger.info(f"   Shape: {vectors.shape}, dtype: {vectors.dtype}")

    if vectors.shape[1] != EXPECTED_DIM:
        logger.warning(
            f"⚠️  Vector dimension {vectors.shape[1]} != expected {EXPECTED_DIM}. "
            "Continuing, but check your embedding model."
        )

    # ─── 3. Separate metadata (drop the embedding column) ────────── #
    metadata = df.drop(columns=[EMBEDDING_COLUMN]).reset_index(drop=True)
    logger.info(f"📋 Metadata columns: {list(metadata.columns)}")

    # ─── 4. Build the FAISS index ────────────────────────────────── #
    logger.info("🏗  Building FAISS index...")
    start = time.time()
    index = FaissIndex(dimension=vectors.shape[1])
    index.build(vectors, metadata, normalize=True)
    elapsed = time.time() - start
    logger.info(f"✅ Index built in {elapsed:.2f}s")

    # ─── 5. Save to disk ─────────────────────────────────────────── #
    logger.info("💾 Saving index and metadata...")
    index.save(INDEX_FILE, METADATA_FILE)

    # ─── 6. Stats ─────────────────────────────────────────────────── #
    stats = index.stats()
    logger.info("📊 Index statistics:")
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")

    # ─── 7. Sanity check: self-query should return rank-1 self ───── #
    logger.info("🔍 Sanity check: querying index with its own row-0 vector...")
    probe = vectors[0:1]  # (1, D)
    results = index.search(probe, k=1)
    if results:
        top = results[0]
        if top["row"] == 0:
            logger.info("✅ Self-query returned the correct chunk at rank 1.")
        else:
            logger.warning(
                f"⚠️  Self-query returned row {top['row']} instead of 0. "
                "Check normalization and vector alignment."
            )

    logger.info("=" * 60)
    logger.info("✅ Indexing complete!")
    logger.info(f"   Index file:    {INDEX_FILE}")
    logger.info(f"   Metadata file: {METADATA_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
