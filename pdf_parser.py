"""PDF text extraction using PyMuPDF"""
import fitz  # PyMuPDF
from typing import Tuple


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails"""
    pass


class PDFParser:
    """Extract text from PDF files using PyMuPDF"""
    
    def parse(self, file_bytes: bytes) -> Tuple[str, int]:
        """
        Extract text from PDF.
        
        Args:
            file_bytes: Raw PDF file content
            
        Returns:
            (extracted_text, page_count)
            
        Raises:
            PDFExtractionError: If extraction fails
        """
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            # Extract text from all pages
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_parts.append(page.get_text())
            
            # Combine all text
            full_text = "\n".join(text_parts)
            page_count = len(doc)
            
            doc.close()
            
            return full_text, page_count
            
        except Exception as e:
            # Wrap any exception as PDFExtractionError
            raise PDFExtractionError(f"Failed to extract text from PDF: {str(e)}") from e
