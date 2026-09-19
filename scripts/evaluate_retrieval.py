"""
Run retrieval evaluation against the golden dataset (LLM-as-judge).

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --k 5
    python scripts/evaluate_retrieval.py --golden path/to/golden.jsonl --output run_name.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from evaluation.judge import make_judge
from evaluation.retrieval_eval import (
    RetrievalEvaluator,
    format_per_entry,
    format_report,
    load_golden_dataset,
)
from llm.errors import LLMError
from llm.retry import RetryExhausted
from retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────── #
DEFAULT_GOLDEN_FILE = "data/evaluation/golden_dataset/qa_pairs_with_context.jsonl"
INDEX_FILE = "data/processed/indexes/books_v2.faiss"
METADATA_FILE = "data/processed/indexes/books_v2_meta.parquet"
MODEL_KEY = "bge-small"
JUDGE_MODEL_KEY = "gemini-flash-lite"
RESULTS_DIR = "data/evaluation/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval (LLM-as-judge).")
    parser.add_argument("--k", type=int, default=5, help="Top-k to retrieve.")
    parser.add_argument(
        "--golden",
        type=str,
        default=DEFAULT_GOLDEN_FILE,
        help="Path to golden dataset JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="retrieval_v3.json",
        help="Filename for the saved results JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("📏 RETRIEVAL EVALUATION (LLM-as-judge)")
    logger.info("=" * 70)

    logger.info(f"Loading golden dataset: {args.golden}")
    entries = load_golden_dataset(args.golden)
    logger.info(f"✅ Loaded {len(entries)} golden queries")

    logger.info("Loading retriever...")
    retriever = Retriever(
        index_path=INDEX_FILE,
        metadata_path=METADATA_FILE,
        model_key=MODEL_KEY,
    )

    logger.info(f"Initializing judge ({JUDGE_MODEL_KEY})...")
    judge = make_judge(model_key=JUDGE_MODEL_KEY)

    logger.info(f"Running {len(entries)} queries (k={args.k})...")
    evaluator = RetrievalEvaluator(retriever, judge)

    try:
        results = evaluator.evaluate(entries, k=args.k)
    except (LLMError, RetryExhausted) as e:
        logger.error(f"❌ Evaluation aborted: {e}")
        logger.info("💡 Partial judgments are preserved in the judge cache.")
        logger.info(
            "   Re-run later (after the quota resets) to finish the remaining queries."
        )
        return

    report = format_report(results, k=args.k)
    print(report)
    print(format_per_entry(results))

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output

    summary = {
        "config": {
            "k": args.k,
            "model_key": MODEL_KEY,
            "judge_model_key": JUDGE_MODEL_KEY,
            "index": INDEX_FILE,
            "golden_file": args.golden,
        },
        "overall": RetrievalEvaluator.summarize(results, args.k),
        "per_book": RetrievalEvaluator.breakdown(results, args.k, lambda e: e.book, "book")[
            "groups"
        ],
        "per_difficulty": RetrievalEvaluator.breakdown(
            results, args.k, lambda e: e.difficulty, "difficulty"
        )["groups"],
        "per_query": [
            {
                "id": r.entry.id,
                "book": r.entry.book,
                "difficulty": r.entry.difficulty,
                "query": r.entry.query,
                "hit": r.hit,
                "rank": r.rank,
                "best_score": r.best_score,
                "chunk_judgments": r.chunk_judgments,
                "top_retrieved": [
                    {"book": c.get("book"), "page": c.get("page"), "score": c.get("score")}
                    for c in r.retrieved
                ],
            }
            for r in results
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Results saved to {out_path}")


if __name__ == "__main__":
    main()