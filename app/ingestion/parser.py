import re
from typing import Any

from pydantic import BaseModel

from app.domain.models import ElementType


class ParsedElement(BaseModel):
    page_number: int
    bbox: dict[str, float]       # {x0, y0, x1, y1}
    type: ElementType
    content: str
    summary: str = ""
    image_path: str = ""
    image_hash: str = ""
    caption_status: str = "pending"
    pdf_caption: str = ""
    ocr_applied: bool = False
    ocr_confidence: float | None = None
    image_bytes: bytes | None = None


class ParseResult(BaseModel):
    elements: list[ParsedElement]
    needs_ocr: bool = False
    needs_ocr_pages: list[int] = []
    page_count: int = 0


class PyMuPDFParser:
    """PyMuPDF (fitz) parser with vector chart detection, header/footer stripping, multi-column sorting, and per-page OCR detection."""

    def __init__(
        self,
        header_margin_pct: float = 0.05,
        footer_margin_pct: float = 0.05,
        min_image_size: float = 100.0,
    ):
        self.header_margin_pct = header_margin_pct
        self.footer_margin_pct = footer_margin_pct
        self.min_image_size = min_image_size

    def parse(self, file_bytes: bytes, filename: str = "") -> ParseResult:
        try:
            import fitz  # type: ignore[import-untyped]
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception:
            text = file_bytes.decode("utf-8", errors="ignore")
            if not text.strip():
                return ParseResult(elements=[], needs_ocr=True, needs_ocr_pages=[1], page_count=1)
            return ParseResult(
                elements=[
                    ParsedElement(
                        page_number=1,
                        bbox={"x0": 0.0, "y0": 0.0, "x1": 500.0, "y1": 700.0},
                        type=ElementType.TEXT,
                        content=text,
                    )
                ],
                needs_ocr=False,
                needs_ocr_pages=[],
                page_count=1,
            )

        elements: list[ParsedElement] = []
        needs_ocr_pages: list[int] = []
        total_text_length = 0

        for page_idx, page in enumerate(doc, start=1):
            page_rect = page.rect
            page_height = page_rect.height
            page_width = page_rect.width

            header_limit = page_height * self.header_margin_pct
            footer_limit = page_height * (1.0 - self.footer_margin_pct)

            blocks = page.get_text("blocks")
            page_text_len = 0

            # Sort blocks by column then vertical position
            blocks = sorted(blocks, key=lambda b: (0 if b[0] < page_width / 2 else 1, b[1]))

            text_blocks_on_page = []
            figure_bboxes_on_page = []

            for b in blocks:
                x0, y0, x1, y1, text_content, block_no, block_type = b[:7]

                if y0 < header_limit or y1 > footer_limit:
                    continue

                bbox = {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}

                if block_type == 0:  # Text block
                    cleaned_text = text_content.strip()
                    if not cleaned_text:
                        continue

                    page_text_len += len(cleaned_text)
                    is_heading = len(cleaned_text) < 80 and not cleaned_text.endswith(".")
                    elem_type = ElementType.HEADING if is_heading else ElementType.TEXT

                    elem = ParsedElement(
                        page_number=page_idx,
                        bbox=bbox,
                        type=elem_type,
                        content=cleaned_text,
                    )
                    elements.append(elem)
                    text_blocks_on_page.append(elem)

                elif block_type == 1:  # Raster Image / Figure
                    w = x1 - x0
                    h = y1 - y0
                    # Skip decorative images (< 100px)
                    if w < self.min_image_size or h < self.min_image_size:
                        continue

                    figure_bboxes_on_page.append(bbox)
                    elements.append(
                        ParsedElement(
                            page_number=page_idx,
                            bbox=bbox,
                            type=ElementType.FIGURE,
                            content="[Raster Figure]",
                            caption_status="pending",
                        )
                    )

            # Vector Chart Detection using page.get_drawings()
            try:
                drawings = page.get_drawings()
                vector_clusters = self._cluster_vector_drawings(drawings, page_width, page_height, header_limit, footer_limit)
                for v_bbox in vector_clusters:
                    # Check that it doesn't substantially overlap an existing raster image
                    if not self._overlaps_any(v_bbox, figure_bboxes_on_page):
                        figure_bboxes_on_page.append(v_bbox)
                        elements.append(
                            ParsedElement(
                                page_number=page_idx,
                                bbox=v_bbox,
                                type=ElementType.FIGURE,
                                content="[Vector Chart Figure]",
                                caption_status="pending",
                            )
                        )
            except Exception:
                pass

            # PDF Caption Proximity Matching (associate nearby "Figure N" or "Chart N" text with figure elements)
            for elem in elements:
                if elem.page_number == page_idx and elem.type == ElementType.FIGURE:
                    caption = self._find_nearby_caption(elem.bbox, text_blocks_on_page)
                    if caption:
                        elem.pdf_caption = caption

            total_text_length += page_text_len
            if page_text_len < 20:
                needs_ocr_pages.append(page_idx)

        needs_ocr = len(needs_ocr_pages) > 0 and len(needs_ocr_pages) == len(doc)

        return ParseResult(
            elements=elements,
            needs_ocr=needs_ocr,
            needs_ocr_pages=needs_ocr_pages,
            page_count=len(doc),
        )

    def _cluster_vector_drawings(
        self,
        drawings: list[dict[str, Any]],
        page_width: float,
        page_height: float,
        header_limit: float,
        footer_limit: float,
    ) -> list[dict[str, float]]:
        """Cluster vector drawings into contiguous bounding box regions for vector charts."""
        rects = []
        for d in drawings:
            rect = d.get("rect")
            if not rect:
                continue
            x0, y0, x1, y1 = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
            w = x1 - x0
            h = y1 - y0
            # Include vector path lines/rects, filtering out page-wide background rects
            if (w < 2 and h < 2) or w > page_width * 0.95 or h > page_height * 0.95:
                continue
            if y0 < header_limit or y1 > footer_limit:
                continue
            rects.append((x0, y0, x1, y1))

        if not rects:
            return []

        # Merge overlapping or close vector rectangles (35pt margin for chart elements)
        clusters: list[tuple[float, float, float, float]] = []
        for r in rects:
            merged = False
            for i, c in enumerate(clusters):
                if not (r[2] < c[0] - 35 or r[0] > c[2] + 35 or r[3] < c[1] - 35 or r[1] > c[3] + 35):
                    clusters[i] = (min(c[0], r[0]), min(c[1], r[1]), max(c[2], r[2]), max(c[3], r[3]))
                    merged = True
                    break
            if not merged:
                clusters.append(r)

        # Filter clusters to reasonable chart sizes (width & height >= min_image_size)
        result_bboxes = []
        for c in clusters:
            w = c[2] - c[0]
            h = c[3] - c[1]
            if w >= self.min_image_size and h >= self.min_image_size:
                result_bboxes.append({"x0": c[0], "y0": c[1], "x1": c[2], "y1": c[3]})

        return result_bboxes

    def _overlaps_any(self, bbox: dict[str, float], existing: list[dict[str, float]]) -> bool:
        for ex in existing:
            # Simple bounding box overlap check
            if not (bbox["x1"] <= ex["x0"] or bbox["x0"] >= ex["x1"] or bbox["y1"] <= ex["y0"] or bbox["y0"] >= ex["y1"]):
                return True
        return False

    def _find_nearby_caption(self, fig_bbox: dict[str, float], text_elems: list[ParsedElement]) -> str:
        """Find text element near figure bbox containing 'Figure', 'Chart', or 'Graph'."""
        for t in text_elems:
            # Check proximity within 50 points vertically
            close_vertical = (abs(t.bbox["y0"] - fig_bbox["y1"]) < 50) or (abs(fig_bbox["y0"] - t.bbox["y1"]) < 50)
            if close_vertical and re.search(r"\b(figure|chart|graph|table)\b", t.content, re.IGNORECASE):
                return t.content
        return ""
