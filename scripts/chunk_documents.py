"""
Build retrieval chunks from page-level documents.

Usage:
    python scripts/chunk_documents.py
"""

import json
import logging
from pathlib import Path

from chunking.chunker import DocumentChunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────── #
INPUT_FILE = "data/processed/chunks/documents_v1.json"
OUTPUT_JSONL = "data/processed/chunks/chunks_v2.jsonl"
OUTPUT_PARQUET = "data/processed/chunks/chunks_v2.parquet"

CHUNK_SIZE = 1000
OVERLAP = 200
MIN_CHUNK_SIZE = 100


def main() -> None:
    if not Path(INPUT_FILE).exists():
        logger.error(f"❌ Input file not found: {INPUT_FILE}")
        logger.info("💡 Run ingestion first to create documents_v1.json")
        return

    logger.info("=" * 60)
    logger.info("📦 STARTING DOCUMENT CHUNKING (v2)")
    logger.info("=" * 60)

    logger.info(f"📄 Loading documents from: {INPUT_FILE}")
    with open(INPUT_FILE, encoding="utf-8") as f:
        documents = json.load(f)
    logger.info(f"✅ Loaded {len(documents)} documents")

    if documents:
        sample = documents[0]
        logger.info("\n📝 Sample document:")
        logger.info(f"   ID: {sample.get('id')}")
        logger.info(f"   Book: {sample.get('book_title')}")
        logger.info(f"   Page: {sample.get('page_number')}")
        logger.info(f"   Content length: {len(sample.get('content', ''))} chars")

    logger.info("\n🔧 Initializing chunker...")
    chunker = DocumentChunker(
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
    )

    logger.info("\n🔄 Creating chunks...")
    chunks = chunker.chunk_documents(documents)

    total_chars = sum(len(c.content) for c in chunks)
    avg_size = total_chars / len(chunks) if chunks else 0

    logger.info("\n📊 Chunking Statistics:")
    logger.info(f"   Total chunks: {len(chunks):,}")
    logger.info(f"   Total characters: {total_chars:,}")
    logger.info(f"   Average chunk size: {avg_size:.0f} chars")
    logger.info(f"   Pages → chunks: {len(documents)} → {len(chunks)}")

    if chunks:
        c = chunks[0]
        logger.info("\n📝 Sample chunk:")
        logger.info(f"   ID: {c.id}")
        logger.info(f"   Book: {c.book_title}")
        logger.info(f"   Page: {c.page_number}")
        logger.info(f"   Chunk: {c.chunk_index + 1}/{c.total_chunks}")
        logger.info(f"   Section: {c.section or '(none)'}")
        logger.info(f"   Size: {len(c.content)} chars")
        logger.info(f"   Preview: {c.content[:100]}...")

    logger.info("\n💾 Saving chunks...")
    chunker.save_chunks(chunks, OUTPUT_JSONL)
    chunker.save_chunks_parquet(chunks, OUTPUT_PARQUET)

    logger.info("\n✅ Chunking complete!")
    logger.info(f"📁 JSONL:   {OUTPUT_JSONL}")
    logger.info(f"📁 Parquet: {OUTPUT_PARQUET}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()