"""
Run the chunker on documents_v1.json
Usage: python scripts/chunk_documents.py
"""

import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.chunking.chunker import DocumentChunker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run the chunker"""

    # Configuration
    INPUT_FILE = "data/processed/chunks/documents_v1.json"
    OUTPUT_JSONL = "data/processed/chunks/chunks_v1.jsonl"
    OUTPUT_PARQUET = "data/processed/chunks/chunks_v1.parquet"

    # Chunker configuration (tune these!)
    CHUNK_SIZE = 500  # Characters per chunk
    OVERLAP = 100  # Overlap between chunks
    MIN_CHUNK_SIZE = 50  # Minimum chunk size

    # Check if input exists
    if not Path(INPUT_FILE).exists():
        logger.error(f"❌ Input file not found: {INPUT_FILE}")
        logger.info("💡 Run ingestion first to create documents_v1.json")
        return

    logger.info("=" * 60)
    logger.info("📦 STARTING DOCUMENT CHUNKING")
    logger.info("=" * 60)

    # Load documents
    logger.info(f"📄 Loading documents from: {INPUT_FILE}")
    with open(INPUT_FILE, encoding="utf-8") as f:
        documents = json.load(f)

    logger.info(f"✅ Loaded {len(documents)} documents")

    # Show sample document
    if documents:
        logger.info("\n📝 Sample document:")
        logger.info(f"   ID: {documents[0].get('id')}")
        logger.info(f"   Book: {documents[0].get('book_title')}")
        logger.info(f"   Page: {documents[0].get('page_number')}")
        logger.info(f"   Content length: {len(documents[0].get('content', ''))} chars")
        logger.info(f"   Content preview: {documents[0].get('content', '')[:100]}...")

    # Initialize chunker
    logger.info("\n🔧 Initializing chunker...")
    chunker = DocumentChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP, min_chunk_size=MIN_CHUNK_SIZE)

    # Create chunks
    logger.info("\n🔄 Creating chunks...")
    chunks = chunker.chunk_documents(documents)

    # Show statistics
    total_chars = sum(len(chunk.content) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0

    logger.info("\n📊 Chunking Statistics:")
    logger.info(f"   Total chunks: {len(chunks)}")
    logger.info(f"   Total characters: {total_chars:,}")
    logger.info(f"   Average chunk size: {avg_chunk_size:.1f} chars")
    logger.info(f"   Compression ratio: {len(documents)} pages → {len(chunks)} chunks")

    # Show sample chunk
    if chunks:
        logger.info("\n📝 Sample chunk:")
        logger.info(f"   ID: {chunks[0].id}")
        logger.info(f"   Book: {chunks[0].book_title}")
        logger.info(f"   Page: {chunks[0].page_number}")
        logger.info(f"   Chunk index: {chunks[0].chunk_index}/{chunks[0].total_chunks}")
        logger.info(f"   Content length: {len(chunks[0].content)} chars")
        logger.info(f"   Content preview: {chunks[0].content[:100]}...")

    # Save chunks
    logger.info("\n💾 Saving chunks...")
    chunker.save_chunks(chunks, OUTPUT_JSONL)
    chunker.save_chunks_parquet(chunks, OUTPUT_PARQUET)

    logger.info("\n✅ Chunking complete!")
    logger.info(f"📁 JSONL: {OUTPUT_JSONL}")
    logger.info(f"📁 Parquet: {OUTPUT_PARQUET}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
