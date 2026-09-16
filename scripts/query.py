"""
Run a query against the FAISS index.

Usage:
    python scripts/query.py "your question here"
    python scripts/query.py "your question" --k 10
"""

from __future__ import annotations

import argparse
import logging
import textwrap

from retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────── #
INDEX_FILE = "data/processed/indexes/books_v1.faiss"
METADATA_FILE = "data/processed/indexes/books_v1_meta.parquet"
MODEL_KEY = "bge-small"
DEFAULT_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the RAG index.")
    parser.add_argument("query", type=str, help="The question to search for.")
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of results to return (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=300,
        help="Number of content characters to show per result (default: 300).",
    )
    return parser.parse_args()


def format_result(rank: int, result: dict, preview_chars: int) -> str:
    """Pretty-print a single search result."""
    header = (
        f"[{rank}] score={result['score']:.4f}  "
        f"{result['book']}  p.{result['page']}  "
        f"(chunk_id={result['chunk_id']})"
    )
    preview = result["content"].replace("\n", " ").strip()
    if len(preview) > preview_chars:
        preview = preview[:preview_chars] + "..."
    body = textwrap.fill(preview, width=100, initial_indent="    ", subsequent_indent="    ")
    return f"{header}\n{body}"


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("🔍 RETRIEVAL")
    logger.info("=" * 60)
    logger.info(f"Query: {args.query!r}")
    logger.info(f"k = {args.k}")

    retriever = Retriever(
        index_path=INDEX_FILE,
        metadata_path=METADATA_FILE,
        model_key=MODEL_KEY,
    )

    results = retriever.search(args.query, k=args.k)

    if not results:
        logger.warning("No results returned.")
        return

    print()
    print("=" * 100)
    print(f"Top {len(results)} results for: {args.query}")
    print("=" * 100)
    for rank, result in enumerate(results, start=1):
        print(format_result(rank, result, args.preview_chars))
        print("-" * 100)
    print()


if __name__ == "__main__":
    main()