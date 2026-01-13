"""Tests for PDFParser - Step 2.2, 2.3"""
import pytest
from pdf_parser import PDFParser, PDFExtractionError


class TestPDFParser:
    """Test PDFParser.parse() method"""
    
    def test_parse_valid_pdf(self):
        """Test parsing a valid PDF returns text and page count"""
        # Create a minimal valid PDF in memory
        # PDF header: %PDF-1.4 + minimal structure
        minimal_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td (Hello World) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000206 00000 n \n"
            b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n300\n%%EOF"
        )
        
        parser = PDFParser()
        text, page_count = parser.parse(minimal_pdf)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert page_count > 0
        assert "Hello" in text or "World" in text  # Should extract some text
    
    def test_parse_invalid_pdf_raises_error(self):
        """Test that invalid PDF bytes raise PDFExtractionError"""
        parser = PDFParser()
        
        # Garbage bytes
        garbage = b"not a pdf file at all"
        with pytest.raises(PDFExtractionError):
            parser.parse(garbage)
        
        # Empty bytes
        with pytest.raises(PDFExtractionError):
            parser.parse(b"")
        
        # Invalid PDF header
        invalid_header = b"NOTPDF-1.4\nsome content"
        with pytest.raises(PDFExtractionError):
            parser.parse(invalid_header)
    
    def test_parse_corrupt_pdf_raises_error(self):
        """Test that corrupt PDF raises PDFExtractionError"""
        parser = PDFParser()
        
        # PDF header but corrupt structure
        corrupt_pdf = b"%PDF-1.4\ncorrupt content here"
        with pytest.raises(PDFExtractionError):
            parser.parse(corrupt_pdf)
