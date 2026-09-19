# ML Books Embedding Pipeline

## Overview

This pipeline takes the semantically chunked text from the chunking phase and converts it into high-dimensional vector representations (embeddings). It utilizes caching and batching to optimize generation speed and outputs the vectors along with metadata into a Parquet file for efficient downstream indexing and retrieval.

## 📁 Directory Structure

```text
rag-ml-books/
├── data/
│   ├── processed/
│   │   ├── chunks/
│   │   │   └── chunks_v2.jsonl              # Input chunks
│   │   └── embeddings/
│   │       └── embeddings_v2.parquet        # Output embeddings (Parquet)
│   └── cache/
│       └── embeddings/
│           └── embeddings_cache.pkl         # Persistent embedding cache
├── src/
│   ├── embedding/
│   │   ├── embedder.py                      # Main embedding generator
│   │   └── cache_manager.py                 # Persistent caching system
│   └── catalog/
│       └── embedding_models.py              # Model configuration registry
└── scripts/
    └── generate_embeddings.py               # Main embedding orchestration script
```

## Components

### 1. Model Configuration (`src/catalog/embedding_models.py`)
Centralized registry for supported embedding models.
- **Default Model**: The pipeline currently defaults to **`bge-small`** (`BAAI/bge-small-en-v1.5`), an optimized and highly performant dense retrieval model.

### 2. Embedder (`src/embedding/embedder.py`)
Generates embeddings for text chunks using HuggingFace's `SentenceTransformer`.
- **Batch Processing**: Processes chunks in configurable batch sizes (default 32) to maximize inference speed.
- **Cache Integration**: Integrates directly with the caching system to skip re-computing embeddings for previously processed chunks.
- **Data Export**: Combines the original chunk metadata and content with the newly generated vector embeddings and exports them to a `.parquet` file.

### 3. Cache Manager (`src/embedding/cache_manager.py`)
Ensures efficiency and resumes interrupted runs cleanly.
- **Persistent Storage**: Saves chunk IDs mapped to their vector representations to disk.
- **Cache Hits**: When `Embedder` receives chunks, it first checks the cache; any hit bypasses the inference model entirely.

### 4. Orchestration (`scripts/generate_embeddings.py`)
The main entry point for the embedding process.
- Loads all chunks from the `chunks_vN.jsonl` file.
- Initializes the `Embedder` with the chosen model (`bge-small`), batch size, and caching enabled.
- Measures generation performance (chunks/sec) and outputs progress statistics per book.

**Embedding Schema Example (Parquet Columns):**
```json
{
  "chunk_id": "Unique MD5 hash inherited from the chunking phase",
  "book_title": "Title of the source book",
  "page_number": "The 1-indexed page number of the source document",
  "chunk_index": "The zero-based index of this chunk on the page",
  "content": "The actual text content that was embedded",
  "embedding": "The high-dimensional vector array representing the text (e.g., 384 dimensions for bge-small)",
  "embedding_dim": "The integer dimension count of the vector"
}
```

## Usage

### Run Embedding Generation
```bash
python scripts/generate_embeddings.py
```
This script will read the most recent chunks, identify which ones need to be embedded based on the cache, generate vectors in batches using the `bge-small` model, and save the final output to a Parquet file.
