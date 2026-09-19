"""
Detailed chunk-structure diagnostics.

Reads chunks_v1.jsonl and reports shape, boundaries, overlap, and residue.

Usage:
    python scripts/diagnose_chunks.py
    python scripts/diagnose_chunks.py --chunks data/processed/chunks/chunks_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from hashlib import md5
from pathlib import Path


HEADING_START = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z]")
CHAPTER_START = re.compile(r"^\s*(?:CHAPTER|Chapter)\s+\d+")
PAGE_NUMBER_LINE = re.compile(r"(?:^|\n)\s*\d{1,4}\s*(?=\n|$)")


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def percentile(sorted_list: list[int], p: float) -> int:
    if not sorted_list:
        return 0
    idx = min(int(len(sorted_list) * p), len(sorted_list) - 1)
    return sorted_list[idx]


def starts_clean(content: str) -> bool:
    c = content.lstrip()
    return bool(c) and (c[0].isupper() or c[0] in '"\'([' or c[0].isdigit())


def ends_clean(content: str) -> bool:
    c = content.rstrip()
    return bool(c) and c[-1] in '.?!:;"\''


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk-structure diagnostics.")
    parser.add_argument(
        "--chunks",
        default="data/processed/chunks/chunks_v1.jsonl",
        help="Path to chunks_v1.jsonl",
    )
    args = parser.parse_args()

    path = Path(args.chunks)
    if not path.exists():
        print(f"❌ Not found: {path}")
        return

    print("📊 Analyzing chunks...")
    chunks = load_chunks(path)
    n = len(chunks)
    print(f"\n✅ Total chunks: {n:,}")

    # ─── Book distribution ─────────────────────────────────────────── #
    book_counts = Counter(c["book_title"] for c in chunks)
    print("\n📚 Chunks per book:")
    for book, count in book_counts.most_common():
        print(f"   {book}: {count:,}")

    # ─── Size distribution ─────────────────────────────────────────── #
    sizes = sorted(len(c["content"]) for c in chunks)
    words = sorted(len(c["content"].split()) for c in chunks)
    print("\n📏 Chunk sizes (chars):")
    print(f"   min={sizes[0]}  p10={percentile(sizes, 0.10)}  "
          f"p50={percentile(sizes, 0.50)}  p90={percentile(sizes, 0.90)}  "
          f"p99={percentile(sizes, 0.99)}  max={sizes[-1]}")
    print(f"   mean={sum(sizes)/n:.0f}")
    print("\n📏 Chunk sizes (words):")
    print(f"   min={words[0]}  p50={percentile(words, 0.50)}  "
          f"p90={percentile(words, 0.90)}  max={words[-1]}")
    print(f"   mean={sum(words)/n:.0f}")

    # ─── Chunks per page ──────────────────────────────────────────── #
    by_page: dict[tuple, list[dict]] = {}
    for c in chunks:
        key = (c["book_title"], c["page_number"])
        by_page.setdefault(key, []).append(c)

    counts_per_page = Counter(len(v) for v in by_page.values())
    print(f"\n📄 Chunks per page ({len(by_page):,} unique pages):")
    for chunks_count, pages in sorted(counts_per_page.items()):
        print(f"   {chunks_count} chunk(s): {pages:,} pages")

    # ─── Boundary quality ─────────────────────────────────────────── #
    mid_start = sum(1 for c in chunks if not starts_clean(c["content"]))
    mid_end = sum(1 for c in chunks if not ends_clean(c["content"]))
    print("\n✂️  Boundary quality:")
    print(f"   Start mid-sentence: {mid_start:,} / {n:,} "
          f"({mid_start/n*100:.1f}%)")
    print(f"   End mid-sentence:   {mid_end:,} / {n:,} "
          f"({mid_end/n*100:.1f}%)")

    # ─── Heading / page-number residue ────────────────────────────── #
    heading_hits = sum(1 for c in chunks if HEADING_START.match(c["content"]))
    chapter_hits = sum(1 for c in chunks if CHAPTER_START.match(c["content"]))
    page_num_hits = sum(
        1 for c in chunks if PAGE_NUMBER_LINE.search(c["content"][:150])
    )
    print("\n📚 Residue:")
    print(f"   Start with N.N heading:   {heading_hits:,}")
    print(f"   Start with Chapter N:     {chapter_hits:,}")
    print(f"   Leading page number:      {page_num_hits:,}")

    # ─── Overlap & gaps ───────────────────────────────────────────── #
    overlap_pairs = 0
    gap_pairs = 0
    for key, page_chunks in by_page.items():
        page_chunks.sort(key=lambda x: x["chunk_index"])
        for prev, curr in zip(page_chunks, page_chunks[1:], strict=False):
            if curr["start_char"] < prev["end_char"]:
                overlap_pairs += 1
            if curr["start_char"] > prev["end_char"]:
                gap_pairs += 1

    print("\n🔗 Adjacent-chunk relationships (within a page):")
    print(f"   Pairs with overlap:  {overlap_pairs:,}")
    print(f"   Pairs with gap:      {gap_pairs:,}")

    # ─── Duplicate content ────────────────────────────────────────── #
    hashes = Counter(md5(c["content"].encode()).hexdigest() for c in chunks)
    dup_groups = sum(1 for _, count in hashes.items() if count > 1)
    print(f"\n♻️  Duplicate chunks (exact match): {dup_groups:,}")

    # ─── Sample chunks ────────────────────────────────────────────── #
    print("\n📝 Sample chunks (first 5):")
    for i, chunk in enumerate(chunks[:5]):
        preview = chunk["content"][:150].replace("\n", " ")
        print(f"\n   Chunk {i+1}:")
        print(f"      Book: {chunk['book_title']}")
        print(f"      Page: {chunk['page_number']}")
        print(f"      Chunk: {chunk['chunk_index']+1}/{chunk['total_chunks']}")
        print(f"      Size: {len(chunk['content'])} chars")
        print(f"      Preview: {preview}...")


if __name__ == "__main__":
    main()