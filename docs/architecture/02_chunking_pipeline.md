# ML Books Chunking Pipeline

## Overview

This pipeline processes page-level structured documents from the ingestion phase into smaller, semantically coherent text chunks. It handles hierarchical chunking (paragraphs → sentences → characters), applies overlap, extracts structural boundaries (like headings and page numbers), and outputs data in both JSONL and Parquet formats suitable for generating embeddings.

## 📁 Directory Structure

```text
rag-ml-books/
├── data/
│   ├── processed/
│   │   ├── chunks/
│   │   │   ├── documents_v1.json     # Input from ingestion
│   │   │   ├── chunks_v2.jsonl       # Output chunks (JSONL)
│   │   │   └── chunks_v2.parquet     # Output chunks (Parquet, flattened metadata)
├── src/
│   ├── chunking/
│   │   └── chunker.py              # Main chunking logic and rules
└── scripts/
    ├── chunk_documents.py          # Main chunking orchestration script
    ├── diagnose_chunks.py          # Diagnostic script for chunk stats
    └── inspect_overlap.py          # Tool to verify chunk overlap
```

## 🔧 Components

### 1. Document Chunker (`src/chunking/chunker.py`)
Provides the core logic for splitting documents into retrieval chunks.
- **Hierarchical Splitting**: Paragraphs are the primary semantic unit. Oversized paragraphs are split at sentence boundaries, and oversized sentences are split at character boundaries as a last resort.
- **Heading & Page Number Extraction**: Detects and strips leading chapter/section headings and bare page numbers from the content, saving them into the chunk's metadata rather than the search text.
- **Configurable Overlap**: Merges small pieces up to the target chunk size (default 1000 characters) while maintaining a trailing text overlap (default 200 characters) to preserve context.
- **Page-bound Constraints**: Chunks are constrained to a single page; they never span across multiple pages.

### 2. Orchestration & Diagnostics
- **Chunking Script (`scripts/chunk_documents.py`)**: The main script that loads the ingested documents, applies the `DocumentChunker`, and exports the chunks to both JSONL and Parquet formats.
- **Diagnostics (`scripts/diagnose_chunks.py` & `scripts/inspect_overlap.py`)**: Used to validate chunk sizes, verify overlap integrity, and perform overall health checks on the chunking output.

**Chunk Schema Example:**
```json
{
  "id": "e4f8d9b1c2a0",
  "content": "This is a paragraph about backpropagation...",
  "metadata": {
    "title": "Deep Learning",
    "author": "Ian Goodfellow",
    "page": 83
  },
  "book_title": "Deep Learning",
  "page_number": 83,
  "chunk_index": 0,
  "total_chunks": 3,
  "start_char": 0,
  "end_char": 425,
  "section": "6.5 Back-Propagation",
  "headings": [
    "6.5 Back-Propagation"
  ]
}
```

**Field Descriptions:**
```json
{
  "id": "Unique MD5 hash based on document ID and chunk index.",
  "content": "The specific chunk of text.",
  "metadata": "Original document metadata inherited from ingestion.",
  "book_title": "Title of the source book.",
  "page_number": "The 1-indexed page number of the source document.",
  "chunk_index": "The zero-based index of this chunk on the page.",
  "total_chunks": "Total number of chunks generated for this page.",
  "start_char": "Approximate starting character offset in the cleaned document.",
  "end_char": "Approximate ending character offset in the cleaned document.",
  "section": "The primary detected section or chapter heading.",
  "headings": "A list of all detected headings preceding the chunk."
}
```

## Usage

### Run Chunking
```bash
python scripts/chunk_documents.py
```
This loads the documents, initializes the chunker, generates chunks based on the hierarchical logic, and saves them to disk.

### Chunk Diagnostics
```bash
python scripts/diagnose_chunks.py --chunks data/processed/chunks/chunks_v2.jsonl
python scripts/inspect_overlap.py
```
These scripts analyze the resulting chunk data to ensure overlap rules and chunk size limits are being respected correctly.
