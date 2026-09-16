"""
Detailed extraction diagnostics for RAG retrieval quality.

Reads documents_v1.json and reports metrics that matter for retrieval:
mojibake density, chunk granularity, heading residue, caption survival,
equation density, and per-book breakdowns.

Usage:
    python scripts/diagnose_extraction.py
    python scripts/diagnose_extraction.py --documents path/to/docs.json
    python scripts/diagnose_extraction.py --sample 5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Patterns ──────────────────────────────────────────────────────── #

# Common mojibake sequences produced by PyMuPDF + UTF-8/cp1252 confusion.
# Not exhaustive — but covers the vast majority of what we saw in the
# Hessian test.
MOJIBAKE_PATTERNS = [
    r"Ō[\wąćęłńóśźż]+",       # ŌłÆ, Ōłé, ŌłÜ, ...
    r"Ž[\w]+",                 # ŽĄ, Žā, ...
    r"╬[\w]*",                 # ╬, ...
    r"â€[\w]*",                # â€™, â€œ, ...
    r"Ã[\w]",                  # Ã©, Ã¨, ...
    r"Â[\w]",                  # Â±, Â·
    r"√[\w]",                  # √ó, √∑ (from ×, ÷)
]

# Section heading at start of chunk (after stripping leading whitespace)
HEADING_START = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z]")

# "CHAPTER N" or "Chapter N" at start
CHAPTER_START = re.compile(r"^\s*(?:CHAPTER|Chapter)\s+\d+")

# All-caps heading on its own line
ALLCAPS_HEADING = re.compile(r"^\s*[A-Z][A-Z\s\-]{5,}$", re.MULTILINE)

# Page number alone on a line (1-4 digits, surrounded by newlines)
PAGE_NUMBER_LINE = re.compile(r"(?:^|\n)\s*\d{1,4}\s*(?=\n|$)")

# A page number in the first 80 chars of a chunk (leaked heading pattern)
LEADING_PAGE_NUMBER = re.compile(r"^\s*(?:\S+\s+){0,8}?\b\d{2,4}\b\s+\S")

# Missing space patterns that survived cleaning
MISSING_SPACE = re.compile(r"[a-z][A-Z]|[a-z]\d{2,}")

# Unicode math symbols (broad set)
UNICODE_MATH = re.compile(
    r"[\u2200-\u22FF\u2A00-\u2AFF\u27C0-\u27EF\u2980-\u29FF]"
    r"|[∑∫∏√∂∇∞∈∉⊂⊃∩∪∧∨∀∃¬]"
    r"|[=≠≤≥±×÷]"
)

# Repeated character sequences (letters only)
REPEATED_CHAR = re.compile(r"([a-zA-Z])\1{2,}")


# ─── Analyzer ──────────────────────────────────────────────────────── #


class ExtractionDiagnostics:
    """Detailed diagnostics for retrieval-relevant extraction quality."""

    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents

    # ─── Public API ─────────────────────────────────────────────────── #

    def run(self) -> dict[str, Any]:
        """Compute all diagnostics and return a structured report."""
        return {
            "overall": self._overall(),
            "mojibake": self._mojibake(),
            "chunk_size": self._chunk_size(),
            "headings": self._headings(),
            "captions": self._captions(),
            "equations": self._equations(),
            "text_hygiene": self._text_hygiene(),
            "empty_pages": self._empty_pages(),
            "per_book": self._per_book(),
            "worst_chunks": self._worst_chunks(),
        }

    # ─── Overall ────────────────────────────────────────────────────── #

    def _overall(self) -> dict[str, Any]:
        n = len(self.documents)
        chars = sum(len(d["content"]) for d in self.documents)
        words = sum(len(d["content"].split()) for d in self.documents)
        return {
            "n_documents": n,
            "total_chars": chars,
            "total_words": words,
            "avg_chars_per_chunk": chars / n if n else 0,
            "avg_words_per_chunk": words / n if n else 0,
        }

    # ─── Mojibake ───────────────────────────────────────────────────── #

    def _mojibake(self) -> dict[str, Any]:
        chunk_hits: list[int] = []
        pattern_counts: Counter[str] = Counter()
        affected_patterns: dict[str, list[int]] = defaultdict(list)
        total_mojibake_chars = 0

        for i, doc in enumerate(self.documents):
            content = doc["content"]
            has_mojibake = False
            for pat in MOJIBAKE_PATTERNS:
                matches = re.findall(pat, content)
                if matches:
                    has_mojibake = True
                    pattern_counts[pat] += len(matches)
                    affected_patterns[pat].append(doc.get("page_number", i))
                    total_mojibake_chars += sum(len(m) for m in matches)
            if has_mojibake:
                chunk_hits.append(i)

        n = len(self.documents)
        return {
            "chunks_with_mojibake": len(chunk_hits),
            "chunks_with_mojibake_pct": (len(chunk_hits) / n * 100) if n else 0,
            "total_mojibake_chars": total_mojibake_chars,
            "pattern_counts": dict(pattern_counts.most_common()),
            "patterns_with_examples": {
                pat: affected_patterns[pat][:5] for pat in pattern_counts
            },
        }

    # ─── Chunk size ─────────────────────────────────────────────────── #

    def _chunk_size(self) -> dict[str, Any]:
        lengths = sorted(len(d["content"]) for d in self.documents)
        word_lengths = sorted(len(d["content"].split()) for d in self.documents)

        def pct(sorted_list: list[int], p: float) -> int:
            if not sorted_list:
                return 0
            idx = min(int(len(sorted_list) * p), len(sorted_list) - 1)
            return sorted_list[idx]

        chunks_per_page: Counter[tuple[str, int]] = Counter()
        for d in self.documents:
            key = (d.get("book_title", "?"), d.get("page_number", -1))
            chunks_per_page[key] += 1

        pages_with_multi = sum(1 for v in chunks_per_page.values() if v > 1)
        pages_with_one = sum(1 for v in chunks_per_page.values() if v == 1)

        return {
            "chars_min": lengths[0] if lengths else 0,
            "chars_p50": pct(lengths, 0.50),
            "chars_p90": pct(lengths, 0.90),
            "chars_p99": pct(lengths, 0.99),
            "chars_max": lengths[-1] if lengths else 0,
            "chars_mean": statistics.mean(lengths) if lengths else 0,
            "words_min": word_lengths[0] if word_lengths else 0,
            "words_p50": pct(word_lengths, 0.50),
            "words_p90": pct(word_lengths, 0.90),
            "words_max": word_lengths[-1] if word_lengths else 0,
            "words_mean": statistics.mean(word_lengths) if word_lengths else 0,
            "unique_pages": len(chunks_per_page),
            "pages_with_one_chunk": pages_with_one,
            "pages_with_multiple_chunks": pages_with_multi,
            "max_chunks_per_page": max(chunks_per_page.values()) if chunks_per_page else 0,
        }

    # ─── Headings ───────────────────────────────────────────────────── #

    def _headings(self) -> dict[str, Any]:
        heading_starts = 0
        chapter_starts = 0
        allcaps_starts = 0
        page_number_starts = 0
        samples: dict[str, list[str]] = {
            "heading": [],
            "chapter": [],
            "allcaps": [],
            "page_number_leak": [],
        }

        for doc in self.documents:
            content = doc["content"]
            first_line = content.split("\n", 1)[0] if content else ""

            if HEADING_START.match(content):
                heading_starts += 1
                if len(samples["heading"]) < 5:
                    samples["heading"].append(first_line[:80])

            if CHAPTER_START.match(content):
                chapter_starts += 1
                if len(samples["chapter"]) < 5:
                    samples["chapter"].append(first_line[:80])

            if ALLCAPS_HEADING.match(first_line):
                allcaps_starts += 1
                if len(samples["allcaps"]) < 5:
                    samples["allcaps"].append(first_line[:80])

            if LEADING_PAGE_NUMBER.match(content[:120]):
                page_number_starts += 1
                if len(samples["page_number_leak"]) < 5:
                    samples["page_number_leak"].append(content[:80])

        n = len(self.documents)
        return {
            "chunks_starting_with_heading": heading_starts,
            "chunks_starting_with_chapter": chapter_starts,
            "chunks_starting_with_allcaps": allcaps_starts,
            "chunks_with_leading_page_number": page_number_starts,
            "total_chunks": n,
            "samples": samples,
        }

    # ─── Captions ───────────────────────────────────────────────────── #

    def _captions(self) -> dict[str, Any]:
        figure_re = re.compile(r"\b(Figure|Fig\.)\s*\d+", re.IGNORECASE)
        table_re = re.compile(r"\bTable\s*\d+", re.IGNORECASE)
        algorithm_re = re.compile(r"\b(Algorithm|Alg\.)\s*\d+", re.IGNORECASE)
        equation_re = re.compile(r"\b(Equation|Eq\.)\s*\d+", re.IGNORECASE)

        counts: Counter[str] = Counter()
        for doc in self.documents:
            c = doc["content"]
            counts["figure"] += len(figure_re.findall(c))
            counts["table"] += len(table_re.findall(c))
            counts["algorithm"] += len(algorithm_re.findall(c))
            counts["equation"] += len(equation_re.findall(c))

        return dict(counts)

    # ─── Equations ──────────────────────────────────────────────────── #

    def _equations(self) -> dict[str, Any]:
        # Count chunks with a high density of Unicode math symbols.
        # This is a proxy for "this chunk contains equations."
        chunks_with_math = 0
        high_density_chunks = 0
        total_math_chars = 0

        for doc in self.documents:
            c = doc["content"]
            math_chars = len(UNICODE_MATH.findall(c))
            if math_chars > 0:
                chunks_with_math += 1
                total_math_chars += math_chars
                if len(c) > 0 and (math_chars / len(c)) > 0.03:
                    high_density_chunks += 1

        n = len(self.documents)
        return {
            "chunks_with_math_symbols": chunks_with_math,
            "chunks_with_math_symbols_pct": (chunks_with_math / n * 100) if n else 0,
            "chunks_with_high_math_density": high_density_chunks,
            "total_math_symbols": total_math_chars,
        }

    # ─── Text hygiene ───────────────────────────────────────────────── #

    def _text_hygiene(self) -> dict[str, Any]:
        repeated_hits = 0
        missing_space_hits = 0
        trailing_ws_hits = 0

        for doc in self.documents:
            c = doc["content"]
            if REPEATED_CHAR.search(c):
                repeated_hits += 1
            if MISSING_SPACE.search(c):
                missing_space_hits += 1
            if re.search(r"[ \t]+\n", c):
                trailing_ws_hits += 1

        n = len(self.documents)
        return {
            "chunks_with_repeated_chars": repeated_hits,
            "chunks_with_missing_spaces": missing_space_hits,
            "chunks_with_trailing_whitespace": trailing_ws_hits,
            "chunks_total": n,
        }

    # ─── Empty pages ────────────────────────────────────────────────── #

    def _empty_pages(self) -> dict[str, Any]:
        empties = []
        for d in self.documents:
            if len(d["content"]) < 100:
                empties.append(
                    {
                        "book": d.get("book_title", "?"),
                        "page": d.get("page_number", -1),
                        "chars": len(d["content"]),
                        "n_images": len(d.get("images", [])),
                    }
                )
        return {
            "count": len(empties),
            "details": empties,
        }

    # ─── Per-book ───────────────────────────────────────────────────── #

    def _per_book(self) -> dict[str, Any]:
        books: dict[str, list[dict]] = defaultdict(list)
        for d in self.documents:
            books[d.get("book_title", "?")].append(d)

        out: dict[str, Any] = {}
        for book, docs in books.items():
            mojibake = sum(
                1
                for d in docs
                if any(re.search(p, d["content"]) for p in MOJIBAKE_PATTERNS)
            )
            chars = [len(d["content"]) for d in docs]
            out[book] = {
                "chunks": len(docs),
                "chars_mean": statistics.mean(chars) if chars else 0,
                "chars_max": max(chars) if chars else 0,
                "chunks_with_mojibake": mojibake,
                "chunks_with_mojibake_pct": (mojibake / len(docs) * 100) if docs else 0,
            }
        return out

    # ─── Worst chunks ───────────────────────────────────────────────── #

    def _worst_chunks(self) -> list[dict[str, Any]]:
        """Chunks with the most mojibake — these are the retrieval-killers."""
        scored = []
        for d in self.documents:
            c = d["content"]
            moj = sum(
                len(re.findall(p, c)) for p in MOJIBAKE_PATTERNS
            )
            if moj > 0:
                scored.append(
                    {
                        "book": d.get("book_title", "?"),
                        "page": d.get("page_number", -1),
                        "mojibake_count": moj,
                        "chars": len(c),
                        "preview": c[:120].replace("\n", " "),
                    }
                )
        return sorted(scored, key=lambda x: -x["mojibake_count"])[:20]


# ─── Report ────────────────────────────────────────────────────────── #


def format_report(r: dict[str, Any]) -> str:
    lines: list[str] = []
    sep = "=" * 78

    def add(s: str = "") -> None:
        lines.append(s)

    add(sep)
    add("📊 EXTRACTION DIAGNOSTICS — RETRIEVAL-FOCUSED")
    add(sep)

    # Overall
    o = r["overall"]
    add("")
    add("OVERALL")
    add(f"  Chunks:              {o['n_documents']:,}")
    add(f"  Total chars:         {o['total_chars']:,}")
    add(f"  Total words:         {o['total_words']:,}")
    add(f"  Avg chars/chunk:     {o['avg_chars_per_chunk']:.0f}")
    add(f"  Avg words/chunk:     {o['avg_words_per_chunk']:.0f}")

    # Mojibake
    m = r["mojibake"]
    add("")
    add("🔴 MOJIBAKE  (corrupted math symbols — retrieval killer)")
    add(f"  Chunks with mojibake: {m['chunks_with_mojibake']:,} "
        f"({m['chunks_with_mojibake_pct']:.1f}%)")
    add(f"  Total mojibake chars: {m['total_mojibake_chars']:,}")
    if m["pattern_counts"]:
        add("  Top patterns:")
        for pat, count in list(m["pattern_counts"].items())[:8]:
            add(f"    {pat!r}: {count:,}")

    # Chunk size
    cs = r["chunk_size"]
    add("")
    add("📏 CHUNK SIZE")
    add(f"  chars:  min={cs['chars_min']}  "
        f"p50={cs['chars_p50']}  p90={cs['chars_p90']}  "
        f"p99={cs['chars_p99']}  max={cs['chars_max']}  "
        f"mean={cs['chars_mean']:.0f}")
    add(f"  words:  min={cs['words_min']}  "
        f"p50={cs['words_p50']}  p90={cs['words_p90']}  "
        f"max={cs['words_max']}  mean={cs['words_mean']:.0f}")
    add(f"  Unique pages:           {cs['unique_pages']:,}")
    add(f"  Pages with 1 chunk:     {cs['pages_with_one_chunk']:,}")
    add(f"  Pages with >1 chunk:    {cs['pages_with_multiple_chunks']:,}")
    add(f"  Max chunks on one page: {cs['max_chunks_per_page']}")

    # Headings
    h = r["headings"]
    add("")
    add("📚 HEADING RESIDUE IN CHUNKS")
    add(f"  Start with N.N heading:     {h['chunks_starting_with_heading']:,}")
    add(f"  Start with Chapter N:       {h['chunks_starting_with_chapter']:,}")
    add(f"  Start with ALL-CAPS:        {h['chunks_starting_with_allcaps']:,}")
    add(f"  Leading page number (≤120): {h['chunks_with_leading_page_number']:,}")
    if h["samples"]["page_number_leak"]:
        add("  Example leaks:")
        for s in h["samples"]["page_number_leak"][:3]:
            add(f"    {s!r}")

    # Captions
    c = r["captions"]
    add("")
    add("🖼  CAPTION SURVIVAL")
    add(f"  Figure refs:    {c.get('figure', 0):,}")
    add(f"  Table refs:     {c.get('table', 0):,}")
    add(f"  Algorithm refs: {c.get('algorithm', 0):,}")
    add(f"  Equation refs:  {c.get('equation', 0):,}")

    # Equations
    e = r["equations"]
    add("")
    add("∑  MATH CONTENT")
    add(f"  Chunks w/ math symbols:      {e['chunks_with_math_symbols']:,} "
        f"({e['chunks_with_math_symbols_pct']:.1f}%)")
    add(f"  Chunks w/ high math density: {e['chunks_with_high_math_density']:,}")
    add(f"  Total math symbols:          {e['total_math_symbols']:,}")

    # Text hygiene
    t = r["text_hygiene"]
    add("")
    add("🧹 TEXT HYGIENE")
    add(f"  Chunks w/ repeated chars:      {t['chunks_with_repeated_chars']:,}")
    add(f"  Chunks w/ missing spaces:      {t['chunks_with_missing_spaces']:,}")
    add(f"  Chunks w/ trailing whitespace: {t['chunks_with_trailing_whitespace']:,}")

    # Empty pages
    ep = r["empty_pages"]
    add("")
    add("📄 EMPTY PAGES  (<100 chars)")
    add(f"  Count: {ep['count']}")
    for d in ep["details"][:10]:
        add(f"    {d['book']}  p{d['page']}  chars={d['chars']}  "
            f"images={d['n_images']}")

    # Per book
    add("")
    add("📖 PER BOOK")
    for book, stats in r["per_book"].items():
        add(f"  {book}")
        add(f"    Chunks:               {stats['chunks']:,}")
        add(f"    Chars mean/max:       {stats['chars_mean']:.0f} / "
            f"{stats['chars_max']:,}")
        add(f"    Chunks w/ mojibake:   {stats['chunks_with_mojibake']:,} "
            f"({stats['chunks_with_mojibake_pct']:.1f}%)")

    # Worst chunks
    wc = r["worst_chunks"]
    if wc:
        add("")
        add("🔴 WORST 10 CHUNKS BY MOJIBAKE COUNT")
        add(f"  {'book':<24} {'p':>5} {'mojibake':>9} {'chars':>7}  preview")
        add("  " + "-" * 100)
        for w in wc[:10]:
            add(f"  {w['book'][:22]:<24} {w['page']:>5} "
                f"{w['mojibake_count']:>9} {w['chars']:>7}  {w['preview'][:60]}")

    add("")
    add(sep)
    return "\n".join(lines)


# ─── CLI ───────────────────────────────────────────────────────────── #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose extraction quality.")
    p.add_argument(
        "--documents",
        default="data/processed/chunks/documents_v1.json",
        help="Path to documents_v1.json",
    )
    p.add_argument(
        "--output",
        default="data/processed/chunks/extraction_diagnostics.json",
        help="Where to save the JSON report.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If >0, only analyze first N documents (fast sanity check).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    path = Path(args.documents)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return

    logger.info(f"Loading documents from {path}")
    with open(path, encoding="utf-8") as f:
        documents = json.load(f)
    logger.info(f"Loaded {len(documents):,} documents")

    if args.sample > 0:
        documents = documents[: args.sample]
        logger.info(f"Sampling first {len(documents)} documents")

    diag = ExtractionDiagnostics(documents)
    report = diag.run()
    print(format_report(report))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 JSON report saved to {out}")


if __name__ == "__main__":
    main()