import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pdf_loader import PDFLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create output directories
    Path("data/processed/chunks").mkdir(parents=True, exist_ok=True)
    Path("data/raw/images").mkdir(parents=True, exist_ok=True)
    
    # 1. Load PDFs with image extraction
    logger.info("=" * 60)
    logger.info("📄 STEP 1: Loading PDFs (with image extraction)")
    logger.info("=" * 60)
    
    loader = PDFLoader(
        pdf_dir="data/raw/pdfs",
        metadata_path="data/raw/metadata/books_metadata.json",
        extract_images=True
    )
    documents = loader.load_all_pdfs()
    
    # 2. Save documents
    logger.info("=" * 60)
    logger.info("💾 STEP 2: Saving documents with image references")
    logger.info("=" * 60)
    
    loader.save_to_disk(
        documents, 
        "data/processed/chunks/documents_v1.json"
    )
    
    # 3. Show statistics
    logger.info("=" * 60)
    logger.info("📊 STATISTICS")
    logger.info("=" * 60)
    
    total_pages = len(documents)
    total_words = sum(len(doc.content.split()) for doc in documents)
    total_images = sum(len(doc.images) for doc in documents)
    
    logger.info(f"Total pages: {total_pages:,}")
    logger.info(f"Total words extracted: {total_words:,}")
    logger.info(f"Total images extracted: {total_images:,}")
    logger.info(f"Average words per page: {total_words // total_pages:,}")
    
    # Show per book stats
    book_stats = {}
    for doc in documents:
        book_title = doc.book_title
        if book_title not in book_stats:
            book_stats[book_title] = {'pages': 0, 'words': 0, 'images': 0}
        book_stats[book_title]['pages'] += 1
        book_stats[book_title]['words'] += len(doc.content.split())
        book_stats[book_title]['images'] += len(doc.images)
    
    logger.info("\n📚 Per Book Statistics:")
    for book, stats in book_stats.items():
        logger.info(f"   {book}:")
        logger.info(f"      Pages: {stats['pages']}")
        logger.info(f"      Words: {stats['words']:,}")
        logger.info(f"      Images: {stats['images']}")
    
    logger.info("\n✅ Ingestion complete with image backup!")
    logger.info(f"📁 Images saved to: data/raw/images/")

if __name__ == "__main__":
    main()