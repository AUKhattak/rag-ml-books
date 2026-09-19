# ML Books Indexing Pipeline

## Overview

This pipeline takes the generated embeddings and structures them into a high-performance vector database using FAISS (Facebook AI Similarity Search). It separates the raw vector data from the descriptive metadata, enabling highly optimized, memory-efficient similarity searches using cosine similarity.

## 📁 Directory Structure

```text
rag-ml-books/
├── data/
│   ├── processed/
│   │   ├── embeddings/
│   │   │   └── embeddings_v2.parquet        # Input embeddings + metadata
│   │   └── indexes/
│   │       ├── books_v2.faiss               # Output FAISS binary index
│   │       └── books_v2_meta.parquet        # Output parallel metadata store
├── src/
│   └── indexing/
│       └── faiss_index.py                   # Main FAISS wrapper class
└── scripts/
    └── build_index.py                       # Indexing orchestration script
```

## Components

### 1. FAISS Index Wrapper (`src/indexing/faiss_index.py`)
A custom wrapper around FAISS's `IndexFlatIP` to manage vectors and their corresponding metadata.
- **Normalization & Cosine Similarity**: Embeddings are L2-normalized upon insertion. When querying with normalized vectors, the `IndexFlatIP` (Inner Product) search behaves exactly like cosine similarity.
- **Parallel Metadata Store**: Keeps the metadata (chunk content, page numbers, titles) completely out of the FAISS binary. Row `i` in the FAISS index maps strictly to row `i` in the parallel pandas DataFrame.
- **Persistence**: Handles saving and loading the binary `.faiss` file and the `.parquet` metadata file in tandem.

### 2. Orchestration (`scripts/build_index.py`)
The main script responsible for constructing the final index.
- **Data Loading**: Reads the `.parquet` file output by the embedding pipeline.
- **Data Separation**: Extracts the `embedding` column into an `(N, D)` NumPy float32 array, passing the remaining columns to the metadata DataFrame.
- **Validation**: Verifies the vector dimension aligns with the expected output (384 dimensions for `bge-small`) and executes a quick self-query sanity check at the end to ensure the top-1 result maps to itself.

**Index Schema Layout:**
```text
[FAISS Index (books_v2.faiss)]         [Metadata Store (books_v2_meta.parquet)]
┌─────┬───────────────────────┐        ┌─────┬───────────┬──────────────┬─────────┐
│ Row │ Normalized Vector (D) │        │ Row │ chunk_id  │ book_title   │ content │
├─────┼───────────────────────┤ <====> ├─────┼───────────┼──────────────┼─────────┤
│  0  │ [0.01, -0.05, ...]    │        │  0  │ 9c2a8f... │ Deep Learning│ ...     │
│  1  │ [0.03,  0.11, ...]    │        │  1  │ b381cf... │ Pattern Rec..│ ...     │
└─────┴───────────────────────┘        └─────┴───────────┴──────────────┴─────────┘
```

## Usage

### Run Index Generation
```bash
python scripts/build_index.py
```
This command will load the completed embeddings, separate the high-dimensional vectors from the semantic text, normalize the vectors, build the `IndexFlatIP`, run a sanity check, and output the two required files into the `indexes` directory.
