"""
End-to-end RAG: query → retrieve → generate → print answer.

Usage:
    python scripts/answer.py "What is the difference between L1 and L2 regularization?"
    python scripts/answer.py "..." --k 8
"""

from __future__ import annotations

import argparse
import logging
import textwrap

from generation.generator import Generator
from retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────── #
INDEX_FILE = "data/processed/indexes/books_v2.faiss"
METADATA_FILE = "data/processed/indexes/books_v2_meta.parquet"
EMBEDDING_MODEL_KEY = "bge-small"
GENERATION_MODEL_KEY = "gemini-flash"
DEFAULT_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question against the RAG index.")
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print the retrieved chunks after the answer.",
    )
    return parser.parse_args()


def format_answer(answer: str) -> str:
    """Wrap the answer text for readable terminal output."""
    return textwrap.fill(answer, width=100)


def main() -> None:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("💬 RAG ANSWER")
    logger.info("=" * 70)
    logger.info(f"Query: {args.query!r}")
    logger.info(f"k = {args.k}")

    retriever = Retriever(
        index_path=INDEX_FILE,
        metadata_path=METADATA_FILE,
        model_key=EMBEDDING_MODEL_KEY,
    )

    generator = Generator(
        retriever=retriever,
        model_key=GENERATION_MODEL_KEY,
        top_k=args.k,
    )

    result = generator.answer(args.query)

    print()
    print("=" * 100)
    print("ANSWER")
    print("=" * 100)
    print(format_answer(result.answer))

    if result.citations:
        print()
        print("Sources:")
        for c in result.citations:
            section = f", {c.section}" if c.section else ""
            print(f"  [{c.index}] {c.book}, page {c.page}{section}")

    if args.show_chunks and result.chunks:
        print()
        print("=" * 100)
        print("RETRIEVED CHUNKS")
        print("=" * 100)
        for i, chunk in enumerate(result.chunks, start=1):
            preview = chunk.get("content", "")[:300].replace("\n", " ")
            print(f"\n[{i}] score={chunk.get('score', 0):.4f}  "
                  f"{chunk.get('book', '')}, page {chunk.get('page', '?')}")
            print(textwrap.fill(preview, width=100, initial_indent="    ",
                                subsequent_indent="    "))

    print()


if __name__ == "__main__":
    main()