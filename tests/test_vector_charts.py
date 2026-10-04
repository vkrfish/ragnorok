import os

from app.domain.models import ElementType
from app.ingestion.parser import PyMuPDFParser


def test_vector_chart_detection():
    pdf_path = os.path.join("sample_pdfs", "vector_chart.pdf")
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    parser = PyMuPDFParser(min_image_size=100.0)
    result = parser.parse(file_bytes, "vector_chart.pdf")

    assert result.page_count == 1
    assert not result.needs_ocr

    figure_elements = [e for e in result.elements if e.type == ElementType.FIGURE]
    assert len(figure_elements) >= 1, "Expected at least one vector chart figure region to be detected."

    fig = figure_elements[0]
    assert fig.bbox["x1"] > fig.bbox["x0"]
    assert fig.bbox["y1"] > fig.bbox["y0"]
    # Check PDF caption proximity matching
    assert "Figure 1" in fig.pdf_caption or "Quarterly Sales" in fig.pdf_caption
