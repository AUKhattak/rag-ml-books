## README.md

```markdown
# RAG System for Machine Learning Books

Production-ready RAG system for ML textbooks — ingests PDFs, chunks, embeds, retrieves, generates.

## 📁 Structure

```
rag_ml_books/
├── data/
│   ├── raw/
│   │   ├── pdfs/                     # Input PDFs
│   │   ├── images/                   # Extracted images
│   │   └── metadata/books_metadata.json
│   ├── processed/
│   │   ├── chunks/
│   │   │   ├── documents_v1.json     # Pages
│   │   │   ├── chunks_v1.jsonl       # Chunks (JSONL)
│   │   │   └── chunks_v1.parquet     # Chunks (Parquet)
│   │   └── embeddings/embeddings_v1.parquet
│   └── cache/embeddings/embeddings_cache.pkl
├── src/
│   ├── ingestion/     # pdf_loader.py, chunker.py, metadata_extractor.py
│   ├── embedding/     # embedder.py, model_config.py, cache_manager.py
│   ├── storage/ retrieval/ generation/ evaluation/
├── scripts/
│   ├── run_ingestion.py
│   ├── check_extraction_quality.py
│   ├── chunk_documents.py
│   ├── chunk_verify.py
│   └── test_embeddings.py
├── venv/
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Pipeline
python scripts/run_ingestion.py            # PDFs → documents_v1.json
python scripts/check_extraction_quality.py # Quality report
python scripts/chunk_documents.py          # documents → chunks_v1.jsonl/.parquet
python scripts/chunk_verify.py             # Verify chunks
python scripts/test_embeddings.py          # Test embedder (5 chunks)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `run_ingestion.py` | Extract text + images from PDFs → `documents_v1.json` |
| `check_extraction_quality.py` | Validate extraction quality |
| `chunk_documents.py` | Split pages → `chunks_v1.jsonl` + `.parquet` |
| `chunk_verify.py` | Show chunk stats (count, size, samples) |
| `test_embeddings.py` | Test embedder with 5 chunks + cache |

## Dependencies

```txt
pymupdf>=1.23.0
nltk>=3.8.0
tqdm>=4.66.0
sentence-transformers>=2.2.0
pandas>=2.0.0
pyarrow>=14.0.0
```

Install: `pip install -r requirements.txt`

## Commands

```bash
# Activate env
venv\Scripts\activate

# Run pipeline
python scripts/run_ingestion.py
python scripts/chunk_documents.py
python scripts/generate_embeddings.py

# Inspect data
python -c "import json; d=json.load(open('data/processed/chunks/documents_v1.json')); print(len(d))"
python -c "import json; d=[json.loads(l) for l in open('data/processed/chunks/chunks_v1.jsonl')]; print(len(d))"
python -c "from src.embedding.cache_manager import CacheManager; print(CacheManager().get_stats())"

# Clean
rm -rf data/processed/chunks/* data/cache/embeddings/*
```

## Status

| Stage | Status |
|-------|--------|
| PDF Loading | ✅ 1,548 pages |
| Image Extraction | ✅ Saved |
| Quality Check | ✅ Clean |
| Chunking | ✅ 1,728 chunks |
| Embedding Setup | ✅ Done |
| Embedding Test | 🔄 Next |
| Full Embeddings | ⬜ |
| Vector DB | ⬜ |
| Retrieval | ⬜ |
| Generation | ⬜ |
| Evaluation | ⬜ |

**Dataset:** 2 books · 1,548 pages · 1,728 chunks · avg 2,030 chars/chunk

**Models:** `bge-small` (384d) · `bge-large` (1024d) · `all-mpnet` (768d) · `openai-small` (1536d)

## Next Steps

1. `python scripts/test_embeddings.py`
2. Create `scripts/generate_embeddings.py` → embed all 1,728 chunks
3. Build vector index (FAISS/ChromaDB)
4. Implement retrieval + generation

## Notes

- **Chunking:** 500 chars, 100 overlap, min 50 chars
- **Cache:** MD5(chunk_id) → embedding, saved once per run
- **Cache invalidated** by chunk size or model change

---
