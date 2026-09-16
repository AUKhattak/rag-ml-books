"""
Retrieval evaluation against a golden dataset.

For each golden query, run retrieval and check whether any of the top-k
returned chunks matches the golden (book, page). Reports Recall@k, MRR,
and Precision@k overall, per book, and per difficulty.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


# ─── Data shapes ───────────────────────────────────────────────────── #


@dataclass
class GoldenEntry:
    id: str
    book: str
    query: str
    gold_pages: list[int]
    gold_section: str
    difficulty: str
    expected_answer_summary: str
    notes: str = ""


@dataclass
class QueryResult:
    entry: GoldenEntry
    hit: bool
    rank: int | None  # rank of first correct chunk (1-based), None if not found
    best_score: float
    retrieved: list[dict[str, Any]] = field(default_factory=list)


# ─── Loading ───────────────────────────────────────────────────────── #


def load_golden_dataset(path: str | Path) -> list[GoldenEntry]:
    """Read the JSONL golden dataset into a list of GoldenEntry objects."""
    entries: list[GoldenEntry] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_num}: {e}") from e
            entries.append(
                GoldenEntry(
                    id=obj["id"],
                    book=obj["book"],
                    query=obj["query"],
                    gold_pages=list(obj["gold_pages"]),
                    gold_section=obj.get("gold_section", ""),
                    difficulty=obj["difficulty"],
                    expected_answer_summary=obj.get("expected_answer_summary", ""),
                    notes=obj.get("notes", ""),
                )
            )
    return entries


# ─── Matching ──────────────────────────────────────────────────────── #


def is_match(retrieved: dict[str, Any], entry: GoldenEntry) -> bool:
    """
    A retrieved chunk matches if its book title and page number
    are both in the golden entry's acceptable sets.
    """
    return retrieved.get("book") == entry.book and retrieved.get("page") in entry.gold_pages


# ─── Evaluation ────────────────────────────────────────────────────── #


class RetrievalEvaluator:
    """Runs the golden dataset through a Retriever and computes metrics."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def evaluate(
        self,
        entries: list[GoldenEntry],
        k: int = 5,
    ) -> list[QueryResult]:
        """Run every golden query and record whether the gold page was retrieved."""
        results: list[QueryResult] = []

        for entry in entries:
            retrieved = self.retriever.search(entry.query, k=k)

            rank: int | None = None
            best_score = 0.0
            for r_idx, chunk in enumerate(retrieved, start=1):
                if is_match(chunk, entry):
                    rank = r_idx
                    best_score = chunk.get("score", 0.0)
                    break

            results.append(
                QueryResult(
                    entry=entry,
                    hit=rank is not None,
                    rank=rank,
                    best_score=best_score,
                    retrieved=retrieved,
                )
            )

        return results

    @staticmethod
    def summarize(results: list[QueryResult], k: int) -> dict[str, Any]:
        """Compute aggregate metrics from a list of QueryResults."""
        n = len(results)
        if n == 0:
            return {}

        recall_at_1 = sum(1 for r in results if r.rank == 1) / n
        recall_at_k = sum(1 for r in results if r.hit) / n
        mrr = sum(1.0 / r.rank for r in results if r.rank is not None) / n
        precision_at_k = sum(
            sum(1 for c in r.retrieved if is_match(c, r.entry)) for r in results
        ) / (n * k)

        return {
            "n": n,
            "k": k,
            "recall_at_1": recall_at_1,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "precision_at_k": precision_at_k,
        }

    @staticmethod
    def breakdown(
        results: list[QueryResult],
        k: int,
        key_fn,
        label: str,
    ) -> dict[str, dict[str, Any]]:
        """Group results by an arbitrary key function and summarize each group."""
        groups: dict[str, list[QueryResult]] = {}
        for r in results:
            key = str(key_fn(r.entry))
            groups.setdefault(key, []).append(r)

        return {
            "label": label,
            "groups": {
                name: RetrievalEvaluator.summarize(items, k)
                for name, items in sorted(groups.items())
            },
        }


# ─── Report ────────────────────────────────────────────────────────── #


def format_report(
    results: list[QueryResult],
    k: int,
) -> str:
    """Human-readable report: overall + per book + per difficulty + failures."""
    lines: list[str] = []

    overall = RetrievalEvaluator.summarize(results, k)
    lines.append("=" * 70)
    lines.append(f"RETRIEVAL EVALUATION  (k={k}, n={overall['n']})")
    lines.append("=" * 70)
    lines.append(f"Recall@{1}:     {overall['recall_at_1']:.2%}")
    lines.append(f"Recall@{k}:     {overall['recall_at_k']:.2%}")
    lines.append(f"MRR:           {overall['mrr']:.4f}")
    lines.append(f"Precision@{k}:  {overall['precision_at_k']:.2%}")

    # By book
    by_book = RetrievalEvaluator.breakdown(results, k, lambda e: e.book, "book")
    lines.append("")
    lines.append("Per book:")
    for book, m in by_book["groups"].items():
        lines.append(f"  {book}")
        lines.append(
            f"    Recall@1: {m['recall_at_1']:.2%}   Recall@{k}: {m['recall_at_k']:.2%}   MRR: {m['mrr']:.4f}"
        )

    # By difficulty
    by_diff = RetrievalEvaluator.breakdown(results, k, lambda e: e.difficulty, "difficulty")
    lines.append("")
    lines.append("Per difficulty:")
    for diff in ["easy", "medium", "hard"]:
        if diff not in by_diff["groups"]:
            continue
        m = by_diff["groups"][diff]
        lines.append(f"  {diff:<8} (n={m['n']})")
        lines.append(
            f"    Recall@1: {m['recall_at_1']:.2%}   Recall@{k}: {m['recall_at_k']:.2%}   MRR: {m['mrr']:.4f}"
        )

    # Per-entry failures (only show failures in the report)
    lines.append("")
    lines.append("Failures:")
    failures = [r for r in results if not r.hit]
    if not failures:
        lines.append("  (none)")
    else:
        for r in failures:
            top = r.retrieved[0] if r.retrieved else {}
            lines.append(
                f"  {r.entry.id:<16} "
                f"[{r.entry.difficulty}] "
                f"gold=({r.entry.book[:20]}, p{r.entry.gold_pages}) "
                f"top={top.get('book', '?')[:20]} p{top.get('page', '?')}"
            )

    lines.append("=" * 70)
    return "\n".join(lines)


def format_per_entry(results: list[QueryResult]) -> str:
    """One line per golden query, showing rank of first correct hit."""
    lines: list[str] = []
    lines.append("")
    lines.append("Per-query detail (rank = position of first correct hit):")
    lines.append(f"  {'id':<16} {'diff':<7} {'hit':<5} {'rank':<5} {'score':<7} query")
    lines.append("  " + "-" * 90)
    for r in results:
        rank_str = str(r.rank) if r.rank is not None else "-"
        hit_str = "yes" if r.hit else "no"
        score_str = f"{r.best_score:.3f}" if r.hit else "-"
        lines.append(
            f"  {r.entry.id:<16} {r.entry.difficulty:<7} {hit_str:<5} {rank_str:<5} {score_str:<7} {r.entry.query[:60]}"
        )
    return "\n".join(lines)
