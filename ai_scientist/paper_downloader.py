"""Utilities for downloading and extracting text from academic papers."""

import os
import requests
import time
from typing import Optional, Dict, Tuple

class PaperAccessError(Exception):
    """Raised when paper access/download fails"""
    pass

def download_paper(
    paper_info: Dict,
    output_dir: str,
    timeout: int = 30,
    max_retries: int = 3
) -> Optional[str]:
    """Download paper PDF with fallback mechanisms for different access types.
    
    Args:
        paper_info: Paper metadata from Semantic Scholar API
        output_dir: Directory to save the PDF
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        Path to downloaded PDF if successful, None otherwise
        
    Raises:
        PaperAccessError: If paper cannot be accessed after all attempts
    """
    paper_id = paper_info.get('paperId')
    if not paper_id:
        raise PaperAccessError("No paper ID provided")
        
    pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")
    
    # Check if already downloaded
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        return pdf_path
        
    # Get PDF access info
    pdf_info = paper_info.get('openAccessPdf', {})
    pdf_status = pdf_info.get('status')
    pdf_url = pdf_info.get('url')
    
    if not pdf_url:
        raise PaperAccessError(f"No PDF URL available (status: {pdf_status})")
        
    # Try downloading with retries
    for attempt in range(max_retries):
        try:
            response = requests.get(pdf_url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Verify content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' not in content_type and 'octet-stream' not in content_type:
                raise PaperAccessError(f"Invalid content type: {content_type}")
                
            # Save PDF
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
                
            if os.path.getsize(pdf_path) < 1000:  # Minimum valid PDF size
                os.remove(pdf_path)
                raise PaperAccessError("Downloaded file too small to be valid PDF")
                
            return pdf_path
            
        except (requests.RequestException, PaperAccessError) as e:
            if attempt == max_retries - 1:
                raise PaperAccessError(f"Failed to download PDF after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

def extract_metadata(paper_info: Dict) -> Dict[str, str]:
    """Extract available metadata when full text isn't accessible."""
    metadata = {
        'title': paper_info.get('title', ''),
        'authors': [a.get('name', '') for a in paper_info.get('authors', [])],
        'year': paper_info.get('year'),
        'venue': paper_info.get('venue', ''),
        'abstract': paper_info.get('abstract', ''),
        'citation_count': paper_info.get('citationCount', 0),
        'pdf_status': paper_info.get('openAccessPdf', {}).get('status'),
    }
    return metadata
