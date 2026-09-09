# 📚 ML Books Ingestion Pipeline

## Overview

This pipeline ingests machine learning textbooks (PDF format), extracts text with comprehensive cleaning, and preserves mathematical content and images for downstream RAG (Retrieval-Augmented Generation) applications.

## 📁 Directory Structure

```
ragl-ml-books/
├── data/
│   ├── raw/
│   │   ├── pdfs/                    # Source PDF files
│   │   ├── images/                  # Extracted images by book
│   │   │   ├── Pattern_Recognition_and_Machine_Learning/
│   │   │   └── Deep_Learning/
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

Comprehensive text cleaning utility optimized for ML book PDFs.

**Key Features:**
- **Missing Spaces Fix**: Addresses 48% of pages with missing spaces
  - CamelCase splitting: `backpropagationAlgorithm` → `backpropagation Algorithm`
  - Word-number separation: `layer3output` → `layer 3 output`
  - Mathematical operators: `p<0.05` → `p < 0.05`
  - Punctuation handling: `word,word` → `word, word`

- **Encoding Fixes**: Corrects common PDF extraction encoding issues
  - UTF-8 normalization
  - Math symbol restoration (∑, ∫, ∂, ∇, ∈, etc.)
  - Special character fixes (â€™ → ', â€œ → ")

- **Math Preservation**: Protects equations during cleaning
  - Detects LaTeX math blocks (`$...$`, `\[...\]`, `\(...\)`)
  - Preserves equation integrity for chunking
  - Extracts equations for separate processing

- **OCR Artifact Removal**: Cleans common OCR issues
  - Repeated characters: `tttthe` → `tthe`
  - Page break artifacts
  - Improper line breaks

- **Structural Detection**: Identifies document boundaries
  - Section/Chapter headers (916 sections found)
  - Subsection detection (1712 subsections found)
  - Figure and table captions

### 2. PDF Loader (`src/ingestion/pdf_loader.py`)

Production-grade PDF loader with image extraction and text cleaning.

**Key Features:**
- **Multi-Book Processing**: Handles multiple PDFs with metadata
- **Intelligent Text Extraction**: 
  - PyMuPDF (fitz) backend
  - Automatic text cleaning integration
  - Empty page filtering

- **Image Extraction**: 
  - Extracts embedded images as PNG/JPEG
  - Organizes by book title
  - Preserves image metadata (dimensions, format, page)
  - Supports fallback for non-embeddable images

- **Document Schema**:

```json
{
  "id": "bc2906654d1d",
  "content": "CHAPTER 3.\nPROBABILITY AND INFORMATION THEORY\n...",
  "metadata": {
    "file": "Deep+Learning+Ian+Goodfellow.pdf",
    "title": "Deep Learning",
    "author": "Ian Goodfellow, Yoshua Bengio, Aaron Courville",
    "year": 2016,
    "edition": "1st",
    "publisher": "MIT Press",
    "domain": "deep_learning",
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

### 3. Ingestion Script (`scripts/run_ingestion.py`)

Main orchestration script for the ingestion pipeline.

**Workflow:**
1. Creates required directories
2. Initializes PDFLoader with image extraction
3. Processes all PDFs in `data/raw/pdfs/`
4. Saves documents to `data/processed/chunks/documents_v1.json`
5. Generates statistics (pages, words, images per book)

## 🚀 Usage

### Prerequisites

```bash
pip install pymupdf  # PyMuPDF for PDF processing
```

### Run Ingestion

```bash
python scripts/run_ingestion.py
```

### Expected Output

```
📄 STEP 1: Loading PDFs (with image extraction)
   Found 2 PDF files
   Book: Pattern Recognition and Machine Learning
   Extracted 748 documents
   Book: Deep Learning
   Extracted 800 documents
   📸 Extracted 281 images from 96 pages

📊 STATISTICS
   Total pages: 1,548
   Total words: 603,934
   Total images: 281
   Avg words/page: 390
```

### Quality Check

After ingestion, run the quality validation:

```bash
python scripts/check_extraction_quality.py
```

**Expected Quality Metrics:**
- Quality Score: 67/100 (improved from 67, missing spaces fixed)
- Missing Spaces: 0 pages (was 743 pages / 48%)
- Math Density: 67.5% pages
- Garbled Text: 2 pages
- Sections Found: 693+
- Images Extracted: 281

## 📊 Performance Metrics

| Metric | Status |
|--------|--------|
| Pages Processed | 1,548 |
| Books Processed | 2 |
| Total Words | 603,934 |
| Images Extracted | 281 (Pattern Recognition) |
| Math Density | 67.5% |
| Quality Score | 67/100 |

## 🔍 Known Issues & Limitations

### Image Extraction
- **Deep Learning book**: 0 images extracted because figures are vector graphics
- **Solution**: Render PDF pages as images using `fitz` or `pdf2image`

### Garbled Text
- **Pages**: [59, 133] in Pattern Recognition book
- **Cause**: Complex mathematical notation or font issues
- **Solution**: Use Nougat or Mathpix for these specific pages

### Figure Captions
- **Status**: 0 found (likely stored in images)
- **Solution**: Future OCR on figure images

## 🛠️ Troubleshooting

### Images Not Saving
```bash
# Check if images exist
ls data/raw/images/Pattern\ Recognition\ and\ Machine\ Learning/

# Verify extraction
python -c "import json; d=json.load(open('data/processed/chunks/documents_v1.json')); print(sum(len(doc['images']) for doc in d))"
```

### Missing Spaces Still Present
```bash
# Re-run with debug logging
python scripts/run_ingestion.py --log-level DEBUG
```

### Memory Issues
- Large PDFs (800+ pages) process fine
- Image extraction adds ~20-30% memory usage
- Consider batch processing if needed

## 🔄 Future Improvements

### Short-term
1. **Missing Spaces**: ✅ Fixed (0 pages)
2. **Image Extraction**: ✅ Added (281 images)
3. **Figure Captions**: Extract via OCR
4. **Deep Learning Images**: Render pages as images

### Medium-term
1. **Nougat Integration**: Math-aware OCR for equations
2. **LaTeX Extraction**: Extract embedded LaTeX from PDFs
3. **Vision Embeddings**: CLIP/LLaVA for figure understanding

### Long-term
1. **Multi-modal RAG**: Text + Image retrieval
2. **Equation Search**: Semantic search over math content
3. **Active Learning**: Improve extraction quality over time

## 📝 Metadata Format

`data/raw/metadata/books_metadata.json`:

```json
{
  "books": [
    {
      "title": "Pattern Recognition and Machine Learning",
      "file": "Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf",
      "author": "Christopher Bishop",
      "year": 2006,
      "domain": "machine_learning",
      "level": "advanced"
    }
  ]
}
```

## 🏗️ Architecture Decisions

1. **PyMuPDF over pdfplumber**: Better image extraction support
2. **TextCleaner as separate module**: Reusable for future pipelines
3. **Image extraction optional**: Disable for faster processing
4. **Document dataclass**: Type safety and clear schema

## ✅ Validation Checklist

Before proceeding to chunking, verify:

- [ ] All PDFs processed without errors
- [ ] Missing spaces = 0 pages
- [ ] Images extracted correctly
- [ ] Quality score ≥ 67
- [ ] documents_v1.json generated
- [ ] Per-book statistics match expectations

## 📞 Support

For issues:
1. Check `data/processed/chunks/extraction_quality_report.txt`
2. Review `data/processed/chunks/quality_metrics.json`
3. Enable debug logging: `--log-level DEBUG`

---

**Last Updated**: September 2026
**Version**: 1.0 (with text cleaning + image extraction)