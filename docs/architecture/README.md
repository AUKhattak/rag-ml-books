# Architecture Overview

This directory contains the detailed architectural documentation for the RAG (Retrieval-Augmented Generation) pipeline for Machine Learning Books. 

## Overall Architecture

The RAG system is divided into sequential stages. Data flows from raw PDFs to structured, searchable embeddings, which are then queried to generate answers.

```text
Raw PDFs & Metadata
  │
  ▼  [01 Ingestion]       (Cleans text, preserves math, extracts images)
documents_v1.json         (One structured record per PDF page)
  │
  ▼  [02 Chunking]        (Hierarchical split: Paragraphs → Sentences → Chars)
chunks_vN.jsonl           (Semantic retrieval units with preserved metadata)
  │
  ▼  [03 Embedding]       (Dense vector generation via SentenceTransformers)
embeddings_vN.parquet     (Vector arrays + original metadata)
  │
  ▼  [04 Indexing]        (Optimized vector storage for fast cosine similarity)
books_vN.faiss + meta     (FAISS binary index + parallel parquet metadata store)
  │
  ▼  [05 Retrieval]       (Top-k similarity search for embedded user queries)
top-k chunks              (Ranked contextual context for a given query)
  │
  ├──► [06 Generation]    (Constructs prompt with context & queries LLM)
  │    Answer & Citations (Direct answer with exact book/page references)
  │
  ▼    [Evaluation]       (Offline metrics and validation)
recall@k / MRR            (LLM-as-judge against golden dataset)
```

## Detailed Pipeline Stages

To read more about the specific implementation details, schemas, and configurations of each stage, click the links below:

### 1. [Ingestion Pipeline](01_ingestion_pipeline.md)
Handles the loading of raw PDFs, comprehensive text cleaning, missing spaces fixing, and image extraction.

### 2. [Chunking Pipeline](02_chunking_pipeline.md)
Splits the page-level documents hierarchically (paragraphs → sentences → characters) into semantically coherent retrieval chunks with metadata preservation.

### 3. [Embedding Pipeline](03_embedding_pipeline.md)
Converts the text chunks into high-dimensional vector representations using SentenceTransformers (e.g., `bge-small`) with batching and caching support.

### 4. [Indexing Pipeline](04_indexing_pipeline.md)
Builds a highly optimized FAISS vector database alongside a parallel metadata store to enable lightning-fast cosine similarity searches.

### 5. [Retrieval Pipeline](05_retrieval_pipeline.md)
Embeds natural language queries and performs a similarity search against the FAISS index to retrieve the top matching text passages.

### 6. [Generation Pipeline](06_generation_pipeline.md)
Orchestrates the entire RAG flow, combining the retrieved context with the user query in a robust prompt, and queries an LLM (Gemini) to generate precise answers with accurate citations.

---
