"""
PARTE 1: Document Ingestion and Processing Pipeline

This module handles:
- Document extraction (PDF, HTML, text)
- Text normalization and cleaning
- Chunking with overlap
- Metadata extraction
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import PyPDF2
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm

from .config import settings


@dataclass
class DocumentChunk:
    """Represents a chunk of processed document"""
    chunk_id: int
    text: str
    metadata: Dict[str, Any]
    source: str
    page: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class DocumentProcessor:
    """
    Handles document ingestion, cleaning, and chunking
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize document processor

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def extract_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        except Exception as e:
            raise ValueError(f"Error extracting PDF: {str(e)}")

        return text

    def extract_from_url(self, url: str) -> str:
        """
        Extract text from URL (HTML)

        Args:
            url: URL to scrape

        Returns:
            Extracted text
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()
            return text

        except Exception as e:
            raise ValueError(f"Error extracting from URL: {str(e)}")

    def extract_from_text(self, text_path: Path) -> str:
        """
        Extract text from text file

        Args:
            text_path: Path to text file

        Returns:
            File content
        """
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise ValueError(f"Error reading text file: {str(e)}")

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove multiple whitespaces
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]', '', text)

        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)

        # Strip whitespace
        text = text.strip()

        return text

    def create_chunks(self, text: str, source: str = "unknown") -> List[DocumentChunk]:
        """
        Split text into overlapping chunks

        Args:
            text: Text to chunk
            source: Source identifier

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        start = 0
        chunk_id = 0

        # Split by sentences first for better semantic boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence exceeds chunk_size, save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=current_chunk.strip(),
                        metadata={
                            "chunk_size": len(current_chunk),
                            "position": chunk_id
                        },
                        source=source
                    )
                )
                chunk_id += 1

                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_size

        # Add the last chunk
        if current_chunk:
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=current_chunk.strip(),
                    metadata={
                        "chunk_size": len(current_chunk),
                        "position": chunk_id
                    },
                    source=source
                )
            )

        return chunks

    def process_document(self,
                         source_path: str,
                         source_type: str = "auto") -> List[DocumentChunk]:
        """
        Complete pipeline: extract, clean, chunk

        Args:
            source_path: Path or URL to document
            source_type: Type of source (pdf, url, text, auto)

        Returns:
            List of processed chunks
        """
        print(f"Processing document: {source_path}")

        # Auto-detect source type
        if source_type == "auto":
            if source_path.startswith("http"):
                source_type = "url"
            elif source_path.endswith(".pdf"):
                source_type = "pdf"
            else:
                source_type = "text"

        # Extract text
        print(f"Extracting text from {source_type}...")
        if source_type == "pdf":
            raw_text = self.extract_from_pdf(Path(source_path))
        elif source_type == "url":
            raw_text = self.extract_from_url(source_path)
        else:
            raw_text = self.extract_from_text(Path(source_path))

        print(f"Extracted {len(raw_text)} characters")

        # Clean text
        print("Cleaning text...")
        clean_text = self.clean_text(raw_text)
        print(f"Cleaned text: {len(clean_text)} characters")

        # Create chunks
        print("Creating chunks...")
        chunks = self.create_chunks(clean_text, source=source_path)
        print(f"Created {len(chunks)} chunks")

        return chunks

    def save_chunks(self, chunks: List[DocumentChunk], output_path: Path = None):
        """
        Save chunks to JSON file

        Args:
            chunks: List of chunks to save
            output_path: Output file path
        """
        if output_path is None:
            output_path = settings.processed_data_dir / "chunks.json"

        chunks_dict = [chunk.to_dict() for chunk in chunks]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_dict, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: Path = None) -> List[DocumentChunk]:
        """
        Load chunks from JSON file

        Args:
            input_path: Input file path

        Returns:
            List of DocumentChunk objects
        """
        if input_path is None:
            input_path = settings.processed_data_dir / "chunks.json"

        with open(input_path, 'r', encoding='utf-8') as f:
            chunks_dict = json.load(f)

        chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_dict]
        print(f"Loaded {len(chunks)} chunks from {input_path}")

        return chunks
