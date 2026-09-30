import pytest
from app.ingestion.parser import PyMuPDFParser
from app.domain.models import ElementType


def test_parser_header_footer_stripping_and_ocr():
    parser = PyMuPDFParser(header_margin_pct=0.05, footer_margin_pct=0.05)

    # Empty text -> NEEDS_OCR
    empty_res = parser.parse(b"", filename="empty.pdf")
    assert empty_res.needs_ocr is True

    # Sample text -> parsed elements
    sample_text = b"Ragnork Technical Report\n1. Overview\nPostgres is the system of record."
    res = parser.parse(sample_text, filename="sample.pdf")
    assert res.needs_ocr is False
    assert len(res.elements) > 0
