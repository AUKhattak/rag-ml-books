"""
Document chunker for RAG pipeline.

Splits page-level documents into paragraph- and sentence-aware chunks
with heading extraction and configurable size/overlap.

Design:
    - Paragraphs are the primary unit.
    - Oversized paragraphs are split at sentence boundaries.
    - Oversized sentences are split at character boundaries (last resort).
    - Leading headings and page numbers are stripped from content and
      stored in metadata under 'section'.
    - Chunks never span pages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ─── Heading patterns ─────────────────────────────────────────────── #

# "CHAPTER 6" or "Chapter 6" followed by anything on the same line
CHAPTER_RE = re.compile(r"^\s*(?:CHAPTER|Chapter)\s+\d+\b.*$")

# "5.4 The Hessian Matrix" or "5.4.4 Finite differences"
SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{0,120}$")

# A bare page number line: "123" or "xiv"
PAGE_NUMBER_RE = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdmIVXLCDM]{1,7})\s*$")


# ─── Chunk data class ─────────────────────────────────────────────── #


@dataclass
class Chunk:
    """A text chunk with source metadata."""

    id: str
    content: str
    metadata: dict[str, Any]
    book_title: str
    page_number: int
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    section: str = ""
    headings: list[str] = field(default_factory=list)


# ─── Chunker ──────────────────────────────────────────────────────── #


class DocumentChunker:
    """
    Splits documents (one per PDF page) into retrieval chunks.

    Chunks respect paragraph boundaries where possible, fall back to
    sentences, and finally to character windows. Leading headings and
    page numbers are stripped into metadata.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        """
        Args:
            chunk_size: Target character count per chunk.
            overlap: Character overlap between consecutive chunks (within a page).
            min_chunk_size: Chunks smaller than this are dropped.
        """
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

        logger.info("📦 Chunker initialized:")
        logger.info(f"   chunk_size={chunk_size} chars")
        logger.info(f"   overlap={overlap} chars")
        logger.info(f"   min_chunk_size={min_chunk_size} chars")

    # ─── Public API ────────────────────────────────────────────────── #

    def chunk_documents(self, documents: list[dict[str, Any]]) -> list[Chunk]:
        """Split a list of page-level documents into chunks."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self._chunk_single_document(doc))
        logger.info(
            f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents"
        )
        return all_chunks

    def save_chunks(self, chunks: list[Chunk], output_file: str) -> None:
        """Save chunks to a JSONL file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(self._to_dict(chunk), ensure_ascii=False) + "\n")

        logger.info(f"💾 Saved {len(chunks)} chunks to {output_file}")

    def save_chunks_parquet(self, chunks: list[Chunk], output_file: str) -> None:
        """Save chunks to a Parquet file with flattened metadata."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for chunk in chunks:
            row = self._to_dict(chunk)
            row["content_length"] = len(chunk.content)
            row["headings"] = "; ".join(chunk.headings)
            row.pop("metadata", None)  # flattened below
            for k, v in chunk.metadata.items():
                row[f"metadata_{k}"] = v
            rows.append(row)

        pd.DataFrame(rows).to_parquet(output_file, index=False)
        logger.info(f"💾 Saved {len(chunks)} chunks to {output_file} (Parquet)")

    # ─── Single document ──────────────────────────────────────────── #

    def _chunk_single_document(self, doc: dict[str, Any]) -> list[Chunk]:
        content = doc.get("content", "")
        if not content.strip():
            return []

        doc_id = doc.get("id", "unknown")
        metadata = dict(doc.get("metadata", {}))
        book_title = doc.get("book_title", "Unknown")
        page_number = doc.get("page_number", 0)

        # Strip headings / page numbers, keep them in metadata
        content, headings, section = self._strip_headings(content)
        if headings:
            metadata["headings"] = "; ".join(headings)
        if section:
            metadata["section"] = section

        # Split into chunk texts
        chunk_texts = self._split_text(content)

        # Drop chunks that are too small or empty
        kept = [t for t in chunk_texts if len(t.strip()) >= self.min_chunk_size]
        total = len(kept)

        result: list[Chunk] = []
        cursor = 0
        for idx, text in enumerate(kept):
            # Approximate position within cleaned content
            pos = content.find(text, cursor)
            if pos < 0:
                pos = cursor
            start_char = pos
            end_char = pos + len(text)
            cursor = end_char

            result.append(
                Chunk(
                    id=self._generate_chunk_id(doc_id, idx),
                    content=text,
                    metadata=dict(metadata),
                    book_title=book_title,
                    page_number=page_number,
                    chunk_index=idx,
                    total_chunks=total,
                    start_char=start_char,
                    end_char=end_char,
                    section=section,
                    headings=headings,
                )
            )

        return result

    # ─── Heading stripping ────────────────────────────────────────── #

    def _strip_headings(self, text: str) -> tuple[str, list[str], str]:
        """
        Remove leading heading-like lines and page numbers from the top of
        the document.

        Returns:
            (cleaned_text, headings_list, primary_section)
        """
        lines = text.split("\n")
        headings: list[str] = []
        section = ""
        body_start = 0

        # Only scan the first ~10 non-empty lines (headings only appear there)
        seen_content = False
        scanned = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            if seen_content:
                break
            scanned += 1
            if scanned > 12:
                break

            if PAGE_NUMBER_RE.match(stripped):
                body_start = i + 1
                continue

            if CHAPTER_RE.match(stripped):
                headings.append(stripped)
                if not section:
                    section = stripped
                body_start = i + 1
                continue

            if SECTION_RE.match(stripped):
                headings.append(stripped)
                if not section:
                    section = stripped
                body_start = i + 1
                continue

            # First real content line: stop scanning
            seen_content = True

        cleaned = "\n".join(lines[body_start:]).strip()
        return cleaned, headings, section

    # ─── Splitting logic ──────────────────────────────────────────── #

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks.

        Order of preference:
          1. Paragraph-based, then split any oversized paragraph at sentences.
          2. If paragraph splitting produced only garbage (single huge chunk
             with no sentence boundaries), fall back to sentence merge.
          3. Last resort: character window.
        """
        text = text.strip()
        if not text:
            return []

        paragraphs = self._split_by_paragraphs(text)
        chunks = self._merge_paragraphs_to_chunks(paragraphs)

        # If any chunk exceeds 2× the target, the merge couldn't split cleanly.
        # Fall back to whole-text sentence splitting.
        if chunks and max(len(c) for c in chunks) <= self.chunk_size * 2:
            return chunks

        sentences = self._split_by_sentences(text)
        chunks = self._merge_sentences_to_chunks(sentences)

        if chunks and max(len(c) for c in chunks) <= self.chunk_size * 2:
            return chunks

        return self._split_by_chars(text)

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split on blank lines."""
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]

    def _split_by_sentences(self, text: str) -> list[str]:
        """
        Split into sentences.

        Handles: '. ' '! ' '? ' followed by an uppercase letter.
        Also splits on newlines when a paragraph boundary was missed.
        """
        # Treat newlines as potential sentence boundaries too
        text = re.sub(r"\n+", " ", text)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
        return [s.strip() for s in parts if s.strip()]

    # ─── Merge helpers ────────────────────────────────────────────── #

    def _merge_paragraphs_to_chunks(self, paragraphs: list[str]) -> list[str]:
        """Merge paragraphs into chunks; split any paragraph that's too big."""
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            # Oversized single paragraph: flush current, then split it
            if para_size > self.chunk_size:
                if current:
                    chunks.append("\n\n".join(current))
                    current, current_size = [], 0
                chunks.extend(self._split_oversized_paragraph(para))
                continue

            # Normal merge
            if current_size + para_size > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                overlap_text = self._get_overlap_paragraphs(current)
                current = [overlap_text] if overlap_text else []
                current_size = len(overlap_text) if overlap_text else 0

            current.append(para)
            current_size += para_size

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _merge_sentences_to_chunks(self, sentences: list[str]) -> list[str]:
        """Merge sentences into chunks."""
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0

        for sent in sentences:
            # Oversized sentence: flush, then hard-split
            if len(sent) > self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current, current_size = [], 0
                chunks.extend(self._split_by_chars(sent))
                continue

            sent_size = len(sent)
            if current_size + sent_size > self.chunk_size and current:
                chunks.append(" ".join(current))
                overlap_text = self._get_overlap_sentences(current)
                current = [overlap_text] if overlap_text else []
                current_size = len(overlap_text) if overlap_text else 0

            current.append(sent)
            current_size += sent_size

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _split_oversized_paragraph(self, para: str) -> list[str]:
        """Split a single long paragraph using sentences, then chars."""
        sentences = self._split_by_sentences(para)
        if len(sentences) <= 1:
            return self._split_by_chars(para)

        chunks = self._merge_sentences_to_chunks(sentences)
        # If sentence merge still leaves huge chunks (single huge sentence),
        # hard-split them.
        out: list[str] = []
        for c in chunks:
            if len(c) > self.chunk_size * 2:
                out.extend(self._split_by_chars(c))
            else:
                out.append(c)
        return out

    def _split_by_chars(self, text: str) -> list[str]:
        """Last-resort character-window split with overlap."""
        chunks: list[str] = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(text), step):
            piece = text[i : i + self.chunk_size].strip()
            if len(piece) >= self.min_chunk_size:
                chunks.append(piece)
        return chunks

    # ─── Overlap ──────────────────────────────────────────────────── #

    def _get_overlap_paragraphs(self, paragraphs: list[str]) -> str:
        """Take trailing text from paragraphs to seed the next chunk."""
        if not paragraphs:
            return ""
        collected: list[str] = []
        size = 0
        for para in reversed(paragraphs):
            if size + len(para) <= self.overlap:
                collected.insert(0, para)
                size += len(para)
            else:
                remaining = self.overlap - size
                if remaining > 0:
                    collected.insert(0, para[-remaining:])
                break
        return "\n\n".join(collected).strip()

    def _get_overlap_sentences(self, sentences: list[str]) -> str:
        """Take trailing sentences to seed the next chunk."""
        if not sentences:
            return ""
        collected: list[str] = []
        size = 0
        for sent in reversed(sentences):
            if size + len(sent) <= self.overlap:
                collected.insert(0, sent)
                size += len(sent)
            else:
                remaining = self.overlap - size
                if remaining > 0:
                    collected.insert(0, sent[-remaining:])
                break
        return " ".join(collected).strip()

    # ─── IDs & serialization ──────────────────────────────────────── #

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        raw = f"{doc_id}_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _to_dict(chunk: Chunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "book_title": chunk.book_title,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "section": chunk.section,
            "headings": chunk.headings,
        }