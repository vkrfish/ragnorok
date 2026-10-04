import os
import tempfile

from app.core.logging import logger
from app.domain.models import ElementType
from app.ingestion.parser import ParsedElement, ParseResult


class DoclingParser:
    """Docling Parser Adapter implementing layout-aware extraction behind the Parser interface."""

    def __init__(self):
        self._docling_available = False
        try:
            import docling  # type: ignore
            self._docling_available = True
        except ImportError:
            logger.warning("Docling package is not installed. DoclingParser will fallback to PyMuPDF.")

    def parse(self, file_bytes: bytes, filename: str = "") -> ParseResult:
        if not self._docling_available:
            from app.ingestion.parser import PyMuPDFParser
            return PyMuPDFParser().parse(file_bytes, filename)

        # Write to temporary file since Docling requires a file path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            from docling.document_converter import DocumentConverter  # type: ignore
            converter = DocumentConverter()
            result = converter.convert(tmp_path)
            doc = result.document

            elements: list[ParsedElement] = []
            page_count = len(doc.pages) if hasattr(doc, "pages") else 1
            total_text_len = 0
            _needs_ocr_pages: list[int] = []

            for item, level in doc.iterate_items():
                content = getattr(item, "text", "").strip()
                if not content:
                    continue

                page_num = 1
                if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
                    page_num = getattr(item.prov[0], "page_no", 1)

                bbox = {"x0": 0.0, "y0": 0.0, "x1": 500.0, "y1": 700.0}
                if hasattr(item, "prov") and item.prov and hasattr(item.prov[0], "bbox"):
                    b = item.prov[0].bbox
                    bbox = {
                        "x0": float(getattr(b, "l", 0.0)),
                        "y0": float(getattr(b, "t", 0.0)),
                        "x1": float(getattr(b, "r", 500.0)),
                        "y1": float(getattr(b, "b", 700.0)),
                    }

                label = str(getattr(item, "label", "text")).lower()
                if "header" in label or "title" in label:
                    elem_type = ElementType.HEADING
                elif "table" in label:
                    elem_type = ElementType.TABLE
                elif "picture" in label or "figure" in label or "chart" in label:
                    elem_type = ElementType.FIGURE
                else:
                    elem_type = ElementType.TEXT

                elements.append(
                    ParsedElement(
                        page_number=page_num,
                        bbox=bbox,
                        type=elem_type,
                        content=content,
                    )
                )
                total_text_len += len(content)

            return ParseResult(
                elements=elements,
                needs_ocr=total_text_len < 20,
                needs_ocr_pages=[1] if total_text_len < 20 else [],
                page_count=page_count,
            )
        except Exception as e:
            logger.error(f"Docling parsing failed ({e}). Falling back to PyMuPDF.")
            from app.ingestion.parser import PyMuPDFParser
            return PyMuPDFParser().parse(file_bytes, filename)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
