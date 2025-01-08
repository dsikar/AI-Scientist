"""Paper loading utilities for literature review functionality."""

import os
from typing import Optional
import pymupdf
import pymupdf4llm
from pypdf import PdfReader

def load_paper_text(pdf_path: str, num_pages: Optional[int] = None, min_size: int = 100) -> str:
    """Load text from a PDF file using multiple fallback methods.
    
    Args:
        pdf_path: Path to the PDF file
        num_pages: Maximum number of pages to load (None for all pages)
        min_size: Minimum text size to consider valid
        
    Returns:
        Extracted text from the PDF
        
    Raises:
        Exception: If text extraction fails or result is too short
    """
    # Try pymupdf4llm first (best quality)
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) >= min_size:
            return text
        raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")

    # Try pymupdf next
    try:
        doc = pymupdf.open(pdf_path)
        if num_pages:
            doc = doc[:num_pages]
        text = ""
        for page in doc:
            text = text + page.get_text()
        if len(text) >= min_size:
            return text
        raise Exception("Text too short") 
    except Exception as e:
        print(f"Error with pymupdf, falling back to pypdf: {e}")

    # Finally try pypdf
    reader = PdfReader(pdf_path)
    if num_pages is None:
        text = "".join(page.extract_text() for page in reader.pages)
    else:
        text = "".join(page.extract_text() for page in reader.pages[:num_pages])
    if len(text) < min_size:
        raise Exception("Text too short")
        
    return text
