import os

from app.ingestion.docling_parser import DoclingParser


def test_docling_parser_interface():
    pdf_path = os.path.join("sample_pdfs", "sample_report.pdf")
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    parser = DoclingParser()
    result = parser.parse(file_bytes, "sample_report.pdf")

    assert result.page_count >= 1
    assert isinstance(result.elements, list)
