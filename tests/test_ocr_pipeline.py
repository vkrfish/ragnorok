import os

import pytest

from app.adapters.llm import FakeLLMGenerator
from app.ingestion.ocr_processor import OCRProcessor
from app.ingestion.parser import PyMuPDFParser


@pytest.mark.asyncio
async def test_scanned_doc_ocr_detection():
    scanned_path = os.path.join("sample_pdfs", "scanned_doc.pdf")
    with open(scanned_path, "rb") as f:  # noqa: ASYNC230
        file_bytes = f.read()

    parser = PyMuPDFParser()
    result = parser.parse(file_bytes, "scanned_doc.pdf")

    assert result.needs_ocr or len(result.needs_ocr_pages) > 0, "Scanned document should be flagged for OCR."
    assert 1 in result.needs_ocr_pages

    ocr_processor = OCRProcessor(llm_generator=FakeLLMGenerator())
    ocr_elements = await ocr_processor.process_page(file_bytes, page_number=1)

    assert len(ocr_elements) > 0
    assert ocr_elements[0].ocr_applied is True
    assert ocr_elements[0].ocr_confidence is not None


@pytest.mark.asyncio
async def test_mixed_doc_ocr_detection():
    mixed_path = os.path.join("sample_pdfs", "mixed_doc.pdf")
    with open(mixed_path, "rb") as f:  # noqa: ASYNC230
        file_bytes = f.read()

    parser = PyMuPDFParser()
    result = parser.parse(file_bytes, "mixed_doc.pdf")

    assert result.page_count == 2
    # Page 1 is digital text, Page 2 is scanned image
    assert 2 in result.needs_ocr_pages
    assert 1 not in result.needs_ocr_pages
