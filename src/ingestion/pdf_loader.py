import pymupdf
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass, asdict
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a parsed document with metadata"""
    id: str
    content: str
    metadata: Dict[str, Any]
    page_number: int
    book_title: str
    chapter: str
    section: str
    images: List[Dict[str, Any]]

class PDFLoader:
    """Production-grade PDF loader with image extraction and text cleaning"""
    
    def __init__(self, pdf_dir: str, metadata_path: str, extract_images: bool = True):
        self.pdf_dir = Path(pdf_dir)
        self.metadata_path = Path(metadata_path)
        self.extract_images = extract_images
        self.documents = []
        
        # Import text cleaner
        from ..utils.text_cleaner import TextCleaner
        self.text_cleaner = TextCleaner()
        
        # Create image directory if extraction is enabled
        if self.extract_images:
            self.image_root = Path("data/raw/images")
            self.image_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Images will be saved to: {self.image_root}")
    
    def load_all_pdfs(self) -> List[Document]:
        """Load and parse all PDFs in the directory"""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            logger.info(f"📄 Loading: {pdf_path.name}")
            docs = self._load_single_pdf(pdf_path)
            self.documents.extend(docs)
            logger.info(f"   Extracted {len(docs)} documents from {pdf_path.name}")
            
        # Log image extraction stats
        if self.extract_images:
            total_images = sum(len(doc.images) for doc in self.documents)
            pages_with_images = sum(1 for doc in self.documents if doc.images)
            logger.info(f"📸 Extracted {total_images} images from {pages_with_images} pages")
            
        logger.info(f"✅ Loaded {len(self.documents)} total documents from {len(pdf_files)} PDFs")
        return self.documents
    
    def _load_single_pdf(self, pdf_path: Path) -> List[Document]:
        """Load a single PDF and extract structured content with images"""
        docs = []
        doc = pymupdf.open(pdf_path)
        
        # Get book metadata
        book_metadata = self._get_book_metadata(pdf_path.name)
        book_title = book_metadata.get('title', pdf_path.stem)
        logger.info(f"   Book: {book_title}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 1. Extract text with comprehensive cleaning
            text = str(page.get_text())
            
            # Use TextCleaner for all cleaning
            text = self.text_cleaner.clean_text(
                text,
                fix_spaces=True,
                fix_encoding=True,
                fix_math=True,
                normalize_whitespace=True
            )
            
            # Skip empty pages
            if not text.strip():
                continue
            
            # 2. Extract images (if enabled)
            images = []
            if self.extract_images:
                images = self._extract_page_images(doc, page_num, book_title)
            
            doc_id = self._generate_doc_id(pdf_path.name, page_num)
            docs.append(Document(
                id=doc_id,
                content=text,
                metadata={
                    **book_metadata,
                    'page': page_num + 1,
                    'total_pages': len(doc),
                    'book_file': pdf_path.name,
                    'has_images': len(images) > 0,
                    'image_count': len(images),
                },
                page_number=page_num + 1,
                book_title=book_title,
                chapter='',
                section='',
                images=images
            ))
        
        doc.close()
        return docs
    
    def _extract_page_images(self, doc, page_num: int, book_title: str) -> List[Dict[str, Any]]:
        """Extract images from a specific page"""
        images = []
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save the image
                image_path = self._save_image(
                    image_bytes,
                    book_title,
                    page_num,
                    img_index,
                    image_ext
                )
                
                images.append({
                    'path': str(image_path),
                    'xref': xref,
                    'width': base_image.get('width', 0),
                    'height': base_image.get('height', 0),
                    'format': image_ext,
                    'page': page_num + 1,
                })
            except Exception as e:
                logger.warning(f"Failed to extract image on page {page_num + 1}: {e}")
                continue
        
        return images
    
    def _save_image(self, image_bytes: bytes, book_title: str,
                   page_num: int, img_index: int, ext: str) -> Path:
        """Save extracted image to disk"""
        # Create book-specific image directory
        book_dir = self.image_root / self._sanitize_filename(book_title)
        book_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"page_{page_num+1:04d}_img_{img_index+1:03d}.{ext}"
        image_path = book_dir / filename
        
        # Save image
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        return image_path
    
    def _sanitize_filename(self, filename: str) -> str:
        """Remove special characters from filenames"""
        return re.sub(r'[^\w\s-]', '', filename).strip().replace(' ', '_')
    
    def _get_book_metadata(self, filename: str) -> Dict[str, Any]:
        """Get book metadata from the JSON file"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Find the book by filename
            for book in data.get('books', []):
                if book.get('file') == filename:
                    return book
                    
            # Return default if not found
            logger.warning(f"No metadata found for {filename}, using defaults")
            return {
                'title': filename.replace('.pdf', ''),
                'author': 'Unknown',
                'year': 'Unknown',
                'domain': 'machine_learning',
                'level': 'unknown'
            }
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {self.metadata_path}")
            return {
                'title': filename.replace('.pdf', ''),
                'author': 'Unknown',
                'year': 'Unknown',
                'domain': 'machine_learning',
                'level': 'unknown'
            }
    
    def _generate_doc_id(self, filename: str, page: int) -> str:
        """Generate a unique document ID"""
        content = f"{filename}_{page}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def save_to_disk(self, documents: List[Document], output_path: str):
        """Save parsed documents to disk"""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        data = [asdict(doc) for doc in documents]
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"💾 Saved {len(documents)} documents to {out_path}")