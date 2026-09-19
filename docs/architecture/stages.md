# `docs/stages.md`

```markdown
# RAG Pipeline — Stage Documentation

This document describes the ingestion and chunking stages of the RAG pipeline
for ML textbook question answering. It is intended for developers maintaining
or extending the pipeline.

## Pipeline Overview

```
PDFs
  │
  ▼  [Ingestion]
documents_v1.json         (one record per PDF page)
  │
  ▼  [Chunking]
chunks_vN.jsonl           (retrieval units)
  │
  ▼  [Embedding]
embeddings_vN.parquet     (dense vectors + metadata)
  │
  ▼  [Indexing]
books_vN.faiss + meta     (FAISS index + parallel metadata)
  │
  ▼  [Retrieval]
top-k chunks              (for a query)
  │
  ▼  [Evaluation]
recall@k / MRR            (LLM-as-judge against golden dataset)
```

Each stage:

- Has a single class in `src/<stage>/` owning the domain logic.
- Is driven by a thin CLI script in `scripts/`.
- Writes one artifact into `data/processed/` that the next stage consumes.
- Is measured, not assumed.

---

## 1. Ingestion

**Purpose:** Convert PDF pages into structured text records with metadata.

**Input:** `data/raw/pdfs/*.pdf`, `data/raw/metadata/books_metadata.json`
**Output:** `data/processed/chunks/documents_v1.json`

**Owner:** `src/ingestion/pdf_loader.py` — `PDFLoader`
**CLI:** `scripts/run_ingestion.py`

### What it does

1. Opens each PDF with PyMuPDF.
2. Extracts text per page.
3. Runs the raw text through `TextCleaner` (encoding fixes, whitespace
   normalization, common OCR artifacts).
4. Extracts embedded images (optional, saved to `data/raw/images/<book>/`).
5. Attaches book metadata (title, author, domain).
6. Emits one record per page.

### Record shape

```json
{
  "id": "<hash>",
  "content": "cleaned page text",
  "book_title": "Pattern Recognition and Machine Learning",
  "page_number": 208,
  "metadata": { "author": "...", "domain": "...", ... },
  "images": [ ... ]
}
```

### Known limitations

- **Page numbers are PDF indices**, not printed book pages. Printing the
  printed number would require per-book offset tables (see `catalog/`).
- **Figure captions are not extracted.** PyMuPDF's `get_text()` drops text
  rendered inside figure images. Queries about specific figures cannot be
  answered from the current corpus.
- **Front-matter pages** (roman numerals, TOC) are ingested as-is. They
  contribute noise to retrieval but are not removed at this stage.

### Quality check

`scripts/diagnose_extraction.py` reports character distribution, mojibake
density, caption survival, math density, and per-book breakdowns. Run it
after any change to `PDFLoader` or `TextCleaner`.

---

## 2. Chunking

**Purpose:** Split page-level documents into retrieval-sized chunks.

**Input:** `documents_v1.json`
**Output:** `chunks_vN.jsonl`, `chunks_vN.parquet`

**Owner:** `src/chunking/chunker.py` — `DocumentChunker`
**CLI:** `scripts/chunk_documents.py`

### What it does

1. Loads documents (one per page).
2. Strips leading headings and page numbers from each page, storing them
   in `metadata["section"]`.
3. Splits the remaining text into chunks using a tiered strategy:
   - **Paragraph** merge as the primary unit.
   - **Sentence** split for paragraphs that exceed `chunk_size`.
   - **Character window** as a last resort for pathological input.
4. Applies overlap between adjacent chunks within the same page.
5. Drops chunks below `min_chunk_size`.
6. Emits one record per chunk.

### Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `chunk_size` | 1000 | Target characters per chunk |
| `overlap` | 200 | Character overlap between consecutive chunks |
| `min_chunk_size` | 100 | Chunks smaller than this are dropped |

### Invariants

- Chunks **never span pages**. Cross-page continuity is a future feature.
- Chunks **never split inside a sentence** unless the sentence itself
  exceeds `chunk_size`.
- Chunk order and position within the source page are recorded
  (`chunk_index`, `total_chunks`, `start_char`, `end_char`).

### Record shape

```json
{
  "id": "<hash>",
  "content": "chunk text",
  "metadata": { "author": "...", "section": "5.4 The Hessian Matrix" },
  "book_title": "...",
  "page_number": 208,
  "chunk_index": 2,
  "total_chunks": 4,
  "start_char": 1024,
  "end_char": 2180,
  "section": "5.4 The Hessian Matrix",
  "headings": ["5.4 The Hessian Matrix", "249"]
}
```

### Known limitations

- **No math-aware splitting.** Equations that span multiple sentences may
  still be cut. A future version should treat math blocks as atomic.
- **Overlap is character-based**, not semantic. Adjacent chunks share raw
  text, not necessarily a full thought.
- **Section headings are extracted only from the top of a page.** Headings
  mid-page (rare) are left in the content.

### Quality check

`scripts/diagnose_chunks.py` reports size distribution, chunks per page,
boundary cleanliness, heading residue, overlap, gaps, and duplicates. Run
it after any change to `DocumentChunker`.

---

## Conventions

- **Do not modify stages retroactively.** A change to ingestion invalidates
  chunks, embeddings, and the index. Re-run the full chain.
- **Version artifacts by name.** `chunks_v1.jsonl` → `chunks_v2.jsonl` when
  the strategy changes. Same for embeddings and indexes.
- **Measure before optimizing.** Every stage has a diagnostic script. Run
  it, read the numbers, fix the top issue, re-run.
- **Keep the CLI thin.** Business logic lives in `src/`, orchestration in
  `scripts/`.
```
