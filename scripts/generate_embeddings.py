import json
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.embedding.embedder import Embedder

# Silence noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_chunks(file_path: str):
    """Load ALL chunks from JSONL"""
    chunks = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def main():
    CHUNKS_FILE = "data/processed/chunks/chunks_v2.jsonl"
    OUTPUT_FILE = "data/processed/embeddings/embeddings_v2.parquet"
    MODEL_NAME = "bge-small"
    BATCH_SIZE = 32

    logger.info("=" * 60)
    logger.info("🔢 GENERATING EMBEDDINGS FOR ALL CHUNKS")
    logger.info("=" * 60)

    # Load ALL chunks
    logger.info(f"📄 Loading chunks: {CHUNKS_FILE}")
    chunks = load_chunks(CHUNKS_FILE)
    logger.info(f"✅ Loaded {len(chunks)} chunks")

    # Per-book breakdown
    from collections import Counter

    book_counts = Counter(c["book_title"] for c in chunks)
    logger.info("\n📚 Chunks per book:")
    for book, count in book_counts.items():
        logger.info(f"   {book}: {count}")

    # Initialize embedder
    logger.info(f"\n🔧 Initializing embedder ({MODEL_NAME})...")
    embedder = Embedder(model_name=MODEL_NAME, batch_size=BATCH_SIZE, use_cache=True)

    # Generate embeddings
    logger.info(f"\n🔄 Embedding {len(chunks)} chunks...")
    start = time.time()
    embeddings = embedder.embed_chunks(chunks)
    elapsed = time.time() - start

    logger.info(f"\n✅ Done in {elapsed:.1f}s ({len(chunks) / elapsed:.1f} chunks/sec)")

    # Save
    logger.info(f"\n💾 Saving to: {OUTPUT_FILE}")
    embedder.save_embeddings(embeddings, chunks, OUTPUT_FILE)

    logger.info("\n✅ All embeddings generated!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
