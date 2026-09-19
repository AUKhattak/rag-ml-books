# 📚 ML Books Ingestion Pipeline

## Overview

This pipeline ingests machine learning textbooks (PDF format), extracts text with comprehensive cleaning, and preserves mathematical content and images for downstream RAG (Retrieval-Augmented Generation) applications.

## 📁 Directory Structure

```text
rag-ml-books/
├── data/
│   ├── raw/
│   │   ├── pdfs/                    # Source PDF files
│   │   ├── images/                  # Extracted images by book
│   │   └── metadata/
│   │       └── books_metadata.json  # Book metadata
│   └── processed/
│       └── chunks/
│           ├── documents_v1.json    # Extracted documents with metadata
│           ├── extraction_quality_report.txt
│           └── quality_metrics.json
├── src/
│   ├── ingestion/
│   │   └── pdf_loader.py           # Main PDF loader
│   └── utils/
│       └── text_cleaner.py         # Text cleaning utilities
└── scripts/
    ├── run_ingestion.py            # Main ingestion script
    └── check_extraction_quality.py # Quality validation
```

## 🔧 Components

### 1. Text Cleaner (`src/utils/text_cleaner.py`)
Provides comprehensive text cleaning optimized for ML book PDFs.
- **Missing Spaces Fix**: Addresses common PDF extraction issues (e.g., CamelCase splitting, word-number separation).
- **Encoding Fixes**: Corrects UTF-8 normalization and restores math symbols.
- **Math Preservation**: Protects LaTeX math equations and blocks during cleaning to preserve integrity for downstream chunking.
- **Structural Detection**: Identifies document boundaries like section/chapter headers and captions.

### 2. PDF Loader (`src/ingestion/pdf_loader.py`)
Production-grade PDF loader handling text and image extraction.
- **Multi-Book Processing**: Processes multiple PDFs, mapping them to external metadata.
- **Text Extraction**: Uses PyMuPDF backend, integrating with `TextCleaner` while skipping empty pages.
- **Image Extraction**: Extracts embedded images, organizes them by book title, and preserves image metadata (dimensions, format, page number).
- **Document Output**: Generates structured `Document` objects mapping text, associated images, and metadata together.

**Document Schema Example:**
```json
{
  "id": "bc2906654d1d",
  "content": "CHAPTER 3.\nPROBABILITY AND INFORMATION THEORY\n...",
  "metadata": {
    "title": "Deep Learning",
    "author": "Ian Goodfellow, Yoshua Bengio, Aaron Courville",
    "year": 2016,
    "domain": "machine_learning",
    "level": "advanced",
    "page": 83,
    "total_pages": 801,
    "book_file": "Deep+Learning+Ian+Goodfellow.pdf",
    "has_images": false,
    "image_count": 0
  },
  "page_number": 83,
  "book_title": "Deep Learning",
  "chapter": "",
  "section": "",
  "images": []
}
```

**Field Descriptions:**
```json
{
  "id": "Unique MD5 hash based on the filename and page number.",
  "content": "The extracted and cleaned text from the page.",
  "metadata": "Contains book-level attributes (title, author, domain) and page-specific stats (image count).",
  "page_number": "The 1-indexed page number of the PDF.",
  "book_title": "The title derived from the external metadata mapping.",
  "chapter": "Placeholder for structural boundaries.",
  "section": "Placeholder for structural boundaries.",
  "images": "A list of objects containing details (path, size, format) of embedded images found on the page."
}
```


### 3. Orchestration & Quality Assurance
- **Ingestion Script (`scripts/run_ingestion.py`)**: The main entry point. It initializes `PDFLoader`, processes the raw PDFs, saves structured data to `documents_v1.json`, and outputs ingestion statistics.
- **Quality Analyzer (`scripts/check_extraction_quality.py`)**: A validation script that verifies extraction accuracy (text legibility, structural preservation, math density) and dynamically suggests optimal chunking strategies.

## Usage

### Run Ingestion
```bash
python scripts/run_ingestion.py
```
This loads the PDFs, extracts text and images, applies text cleaning, and saves the output structured document file.

### Quality Check
```bash
python scripts/check_extraction_quality.py
```
This assesses the output, generating human-readable and machine-readable quality reports in the processed directory to inform the subsequent chunking stage.
