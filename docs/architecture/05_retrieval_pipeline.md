# ML Books Retrieval Pipeline

## Overview

The retrieval pipeline bridges the gap between natural language questions and the vector database. It takes a user's query, generates an embedding using the identical model used for the chunks, and searches the FAISS index for the most semantically relevant text passages.

## 📁 Directory Structure

```text
rag-ml-books/
├── data/
│   └── processed/
│       └── indexes/
│           ├── books_v2.faiss               # Input FAISS index
│           └── books_v2_meta.parquet        # Input metadata store
├── src/
│   └── retrieval/
│       └── retriever.py                     # Main retrieval logic
└── scripts/
    └── query.py                             # CLI for querying the index
```

## Components

### 1. Retriever (`src/retrieval/retriever.py`)
Encapsulates the logic for loading the index and embedding the query.
- **Model Verification**: Validates that the retrieval embedding model strictly matches the one used to build the index, avoiding dimension mismatches or semantic drift.
- **Query Embedding**: Uses HuggingFace's `SentenceTransformer` (default `bge-small`) to convert the incoming text query into a normalized vector.
- **Vector Search**: Queries the `FaissIndex` for the top-`K` closest matches, joining the FAISS search results (row indices and cosine scores) back with the stored metadata.

### 2. Query Orchestration (`scripts/query.py`)
A standalone CLI tool to independently test and evaluate the retrieval quality before generation.
- Accepts a natural language query and configurable `k` (number of results, default 5).
- Initializes the `Retriever` and executes the search.
- Pretty-prints the ranked results with scores, book titles, page numbers, and truncated text previews.

**Search Result Schema Example:**
```json
[
  {
    "score": 0.8924,
    "row": 421,
    "chunk_id": "9c2a8f3b12",
    "book": "Deep Learning",
    "page": 83,
    "content": "The extracted text of the chunk..."
  }
]
```

## Usage

### Run a Query
```bash
python scripts/query.py "What is backpropagation?" --k 5
```
This script will output the top 5 most relevant chunks from the books, along with their cosine similarity scores and exact source pages.
