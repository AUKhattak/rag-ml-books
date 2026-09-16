"""
Retrieval evaluation against a golden dataset, using an LLM as judge.

For each golden query, run retrieval and ask the judge whether any of the
top-k returned chunks answers the query. Reports Recall@k, MRR, and
Precision@k overall, per book, and per difficulty.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from evaluation.judge import chunk_answers_query
from retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


# ─── Data shapes ───────────────────────────────────────────────────── #


@dataclass
class GoldenEntry:
    id: str
    book: str
    query: str
    difficulty: str
    expected_answer_summary: str
    notes: str = ""


@dataclass
class QueryResult:
    entry: GoldenEntry
    hit: bool
    rank: int | None  # rank of first judged-relevant chunk (1-based)
    best_score: float
    chunk_judgments: list[bool] = field(default_factory=list)
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
                    difficulty=obj["difficulty"],
                    expected_answer_summary=obj.get("expected_answer_summary", ""),
                    notes=obj.get("notes", ""),
                )
            )
    return entries


# ─── Evaluation ────────────────────────────────────────────────────── #


class RetrievalEvaluator:
    """Runs the golden dataset through a Retriever and computes metrics."""

    def __init__(self, retriever: Retriever, judge: Callable[[str], str]):
        self.retriever = retriever
        self.judge = judge

    def evaluate(
        self,
        entries: list[GoldenEntry],
        k: int = 5,
    ) -> list[QueryResult]:
        results: list[QueryResult] = []

        for entry in entries:
            retrieved = self.retriever.search(entry.query, k=k)

            judgments: list[bool] = []
            rank: int | None = None
            best_score = 0.0

            for r_idx, chunk in enumerate(retrieved, start=1):
                is_hit = chunk_answers_query(
                    chunk_content=chunk.get("content", ""),
                    query=entry.query,
                    expected_answer_summary=entry.expected_answer_summary,
                    judge=self.judge,
                )
                judgments.append(is_hit)
                if is_hit and rank is None:
                    rank = r_idx
                    best_score = chunk.get("score", 0.0)

            results.append(
                QueryResult(
                    entry=entry,
                    hit=rank is not None,
                    rank=rank,
                    best_score=best_score,
                    chunk_judgments=judgments,
                    retrieved=retrieved,
                )
            )

        return results

    @staticmethod
    def summarize(results: list[QueryResult], k: int) -> dict[str, Any]:
        n = len(results)
        if n == 0:
            return {}

        recall_at_1 = sum(1 for r in results if r.rank == 1) / n
        recall_at_k = sum(1 for r in results if r.hit) / n
        mrr = sum(1.0 / r.rank for r in results if r.rank is not None) / n

        total_judged = sum(len(r.chunk_judgments) for r in results)
        total_hits = sum(sum(r.chunk_judgments) for r in results)
        precision_at_k = (total_hits / total_judged) if total_judged else 0.0

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
    ) -> dict[str, Any]:
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


def format_report(results: list[QueryResult], k: int) -> str:
    lines: list[str] = []

    overall = RetrievalEvaluator.summarize(results, k)
    lines.append("=" * 70)
    lines.append(f"RETRIEVAL EVALUATION  (k={k}, n={overall['n']})")
    lines.append("=" * 70)
    lines.append(f"Recall@1:      {overall['recall_at_1']:.2%}")
    lines.append(f"Recall@{k}:      {overall['recall_at_k']:.2%}")
    lines.append(f"MRR:           {overall['mrr']:.4f}")
    lines.append(f"Precision@{k}:   {overall['precision_at_k']:.2%}")

    by_book = RetrievalEvaluator.breakdown(results, k, lambda e: e.book, "book")
    lines.append("")
    lines.append("Per book:")
    for book, m in by_book["groups"].items():
        lines.append(f"  {book}")
        lines.append(
            f"    Recall@1: {m['recall_at_1']:.2%}   "
            f"Recall@{k}: {m['recall_at_k']:.2%}   MRR: {m['mrr']:.4f}"
        )

    by_diff = RetrievalEvaluator.breakdown(results, k, lambda e: e.difficulty, "difficulty")
    lines.append("")
    lines.append("Per difficulty:")
    for diff in ["easy", "medium", "hard"]:
        if diff not in by_diff["groups"]:
            continue
        m = by_diff["groups"][diff]
        lines.append(f"  {diff:<8} (n={m['n']})")
        lines.append(
            f"    Recall@1: {m['recall_at_1']:.2%}   "
            f"Recall@{k}: {m['recall_at_k']:.2%}   MRR: {m['mrr']:.4f}"
        )

    lines.append("")
    lines.append("Failures:")
    failures = [r for r in results if not r.hit]
    if not failures:
        lines.append("  (none)")
    else:
        for r in failures:
            top = r.retrieved[0] if r.retrieved else {}
            lines.append(
                f"  {r.entry.id:<16} [{r.entry.difficulty}] "
                f"top={top.get('book', '?')[:22]} p{top.get('page', '?')}"
            )

    lines.append("=" * 70)
    return "\n".join(lines)


def format_per_entry(results: list[QueryResult]) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("Per-query detail (rank = position of first judged-relevant chunk):")
    lines.append(f"  {'id':<16} {'diff':<7} {'hit':<5} {'rank':<5} {'score':<7} query")
    lines.append("  " + "-" * 90)
    for r in results:
        rank_str = str(r.rank) if r.rank is not None else "-"
        hit_str = "yes" if r.hit else "no"
        score_str = f"{r.best_score:.3f}" if r.hit else "-"
        lines.append(
            f"  {r.entry.id:<16} {r.entry.difficulty:<7} {hit_str:<5} "
            f"{rank_str:<5} {score_str:<7} {r.entry.query[:60]}"
        )
    return "\n".join(lines)