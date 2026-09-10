# src/ingestion/chunker.py
"""
Document Chunker for RAG Pipeline
Splits documents into semantic chunks with configurable size and overlap
"""

import json
import hashlib
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    id: str
    content: str
    metadata: Dict[str, Any]
    book_title: str
    page_number: int
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int

class DocumentChunker:
    """
    Splits documents into overlapping chunks for RAG retrieval.
    
    Features:
    - Configurable chunk size and overlap
    - Preserves document metadata
    - Smart splitting at sentence boundaries
    - Handles equations and code blocks
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        min_chunk_size: int = 50,
        respect_sentences: bool = True,
        respect_paragraphs: bool = True
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target number of characters per chunk
            overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size (skip smaller chunks)
            respect_sentences: Try to split at sentence boundaries
            respect_paragraphs: Try to split at paragraph boundaries
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentences = respect_sentences
        self.respect_paragraphs = respect_paragraphs
        
        logger.info(f"📦 Chunker initialized:")
        logger.info(f"   Chunk size: {chunk_size} chars")
        logger.info(f"   Overlap: {overlap} chars")
        logger.info(f"   Min chunk size: {min_chunk_size} chars")
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Split a list of documents into chunks.
        
        Args:
            documents: List of document dicts (from PDFLoader)
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_single_document(doc)
            all_chunks.extend(doc_chunks)
            
        logger.info(f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _chunk_single_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        """
        Split a single document into chunks.
        
        Args:
            doc: Document dict with 'content', 'id', 'metadata', etc.
            
        Returns:
            List of Chunk objects for this document
        """
        content = doc.get('content', '')
        doc_id = doc.get('id', 'unknown')
        metadata = doc.get('metadata', {})
        book_title = doc.get('book_title', 'Unknown')
        page_number = doc.get('page_number', 0)
        
        # Skip empty content
        if not content.strip():
            return []
        
        # Get chunks with smart splitting
        chunks = self._split_text(content)
        
        # Create Chunk objects
        result = []
        for idx, chunk_text in enumerate(chunks):
            # Skip chunks that are too small
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            
            chunk_id = self._generate_chunk_id(doc_id, idx)
            
            chunk = Chunk(
                id=chunk_id,
                content=chunk_text,
                metadata=metadata.copy(),
                book_title=book_title,
                page_number=page_number,
                chunk_index=idx,
                total_chunks=len(chunks),
                start_char=content.find(chunk_text),
                end_char=content.find(chunk_text) + len(chunk_text)
            )
            result.append(chunk)
        
        return result
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using smart boundaries.
        
        Tries to split at:
        1. Paragraph boundaries (if enabled)
        2. Sentence boundaries (if enabled)
        3. Character count (fallback)
        """
        chunks = []
        
        # Clean text
        text = text.strip()
        
        if not text:
            return []
        
        # Try paragraph-based splitting first
        if self.respect_paragraphs:
            paragraphs = self._split_by_paragraphs(text)
            chunks = self._merge_paragraphs_to_chunks(paragraphs)
        
        # If we got chunks from paragraphs, use them
        if chunks:
            return chunks
        
        # Try sentence-based splitting
        if self.respect_sentences:
            sentences = self._split_by_sentences(text)
            chunks = self._merge_sentences_to_chunks(sentences)
        
        # If we got chunks from sentences, use them
        if chunks:
            return chunks
        
        # Fallback: character-based splitting
        chunks = self._split_by_chars(text)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (handles common cases)
        # For ML books, this should work well
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _merge_paragraphs_to_chunks(self, paragraphs: List[str]) -> List[str]:
        """Merge paragraphs into chunks of target size"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If adding this paragraph exceeds chunk size, finalize current chunk
            if current_size + para_size > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_size = len(overlap_text) if overlap_text else 0
            
            current_chunk.append(para)
            current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _merge_sentences_to_chunks(self, sentences: List[str]) -> List[str]:
        """Merge sentences into chunks of target size"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in sentences:
            sent_size = len(sent)
            
            if current_size + sent_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_sentences(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_size = len(overlap_text) if overlap_text else 0
            
            current_chunk.append(sent)
            current_size += sent_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _split_by_chars(self, text: str) -> List[str]:
        """Fallback: Split by character count"""
        chunks = []
        text_len = len(text)
        
        for i in range(0, text_len, self.chunk_size - self.overlap):
            chunk = text[i:i + self.chunk_size]
            if len(chunk.strip()) >= self.min_chunk_size:
                chunks.append(chunk.strip())
        
        return chunks
    
    def _get_overlap(self, paragraphs: List[str]) -> str:
        """Get overlap text from the end of the last paragraphs"""
        if not paragraphs:
            return ''
        
        overlap_text = ''
        current_size = 0
        
        # Work backwards from the end
        for para in reversed(paragraphs):
            para_size = len(para)
            if current_size + para_size <= self.overlap:
                overlap_text = para + '\n\n' + overlap_text
                current_size += para_size
            else:
                # Take part of the paragraph
                chars_to_take = self.overlap - current_size
                if chars_to_take > 0:
                    overlap_text = para[-chars_to_take:] + '\n\n' + overlap_text
                break
        
        return overlap_text.strip()
    
    def _get_overlap_sentences(self, sentences: List[str]) -> str:
        """Get overlap text from the end of the last sentences"""
        if not sentences:
            return ''
        
        overlap_text = ''
        current_size = 0
        
        for sent in reversed(sentences):
            sent_size = len(sent)
            if current_size + sent_size <= self.overlap:
                overlap_text = sent + ' ' + overlap_text
                current_size += sent_size
            else:
                # Take part of the sentence
                chars_to_take = self.overlap - current_size
                if chars_to_take > 0:
                    overlap_text = sent[-chars_to_take:] + ' ' + overlap_text
                break
        
        return overlap_text.strip()
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate unique ID for chunk"""
        content = f"{doc_id}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def save_chunks(self, chunks: List[Chunk], output_file: str):
        """
        Save chunks to JSONL file.
        
        Args:
            chunks: List of Chunk objects
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # Convert to dict
                chunk_dict = {
                    'id': chunk.id,
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'book_title': chunk.book_title,
                    'page_number': chunk.page_number,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                }
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 Saved {len(chunks)} chunks to {output_file}")
    
    def save_chunks_parquet(self, chunks: List[Chunk], output_file: str):
        """
        Save chunks to Parquet file (more efficient).
        
        Args:
            chunks: List of Chunk objects
            output_file: Path to output file
        """
        import pandas as pd
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dicts
        data = []
        for chunk in chunks:
            chunk_dict = {
                'id': chunk.id,
                'content': chunk.content,
                'book_title': chunk.book_title,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'total_chunks': chunk.total_chunks,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char,
                'content_length': len(chunk.content)
            }
            # Flatten metadata
            for key, value in chunk.metadata.items():
                chunk_dict[f'metadata_{key}'] = value
            data.append(chunk_dict)
        
        df = pd.DataFrame(data)
        df.to_parquet(output_file, index=False)
        
        logger.info(f"💾 Saved {len(chunks)} chunks to {output_file} (Parquet)")
