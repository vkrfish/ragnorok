import io
from typing import Any

from PIL import Image

from app.adapters.llm import LLMGenerator, get_llm_generator
from app.core.logging import logger
from app.domain.models import ElementType
from app.domain.schemas import ImagePart
from app.ingestion.parser import ParsedElement


class OCRProcessor:
    """Per-page OCR engine with primary Tesseract (preserving bboxes) and flagged LLM fallback."""

    def __init__(self, llm_generator: LLMGenerator | None = None):
        self.llm_generator = llm_generator or get_llm_generator()

    async def process_page(self, pdf_bytes: bytes, page_number: int) -> list[ParsedElement]:
        """Process a low-text page using Tesseract primary OCR or LLM fallback."""
        pixmap_bytes, width, height = self._render_page_to_png(pdf_bytes, page_number)
        if not pixmap_bytes:
            return []

        # 1. Attempt Primary Tesseract OCR
        tesseract_elements = self._run_tesseract(pixmap_bytes, width, height, page_number)
        if tesseract_elements:
            return tesseract_elements

        # 2. Fallback to LLM OCR if Tesseract unavailable/fails
        logger.warning(f"Tesseract OCR failed/unavailable on page {page_number}. Using LLM OCR fallback.")
        return await self._run_llm_ocr_fallback(pixmap_bytes, page_number)

    def _render_page_to_png(self, pdf_bytes: bytes, page_number: int) -> tuple[bytes | None, float, float]:
        try:
            import fitz  # type: ignore[import-untyped]
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if page_number < 1 or page_number > len(doc):
                return None, 0.0, 0.0
            page = doc[page_number - 1]
            pix = page.get_pixmap(dpi=150)
            return pix.tobytes("png"), float(page.rect.width), float(page.rect.height)
        except Exception as e:
            logger.error(f"Failed to render page {page_number} for OCR: {e}")
            return None, 0.0, 0.0

    def _run_tesseract(
        self,
        png_bytes: bytes,
        pdf_width: float,
        pdf_height: float,
        page_number: int,
    ) -> list[ParsedElement]:
        try:
            import pytesseract  # type: ignore[import-untyped]
            from pytesseract import Output  # type: ignore[import-untyped]

            image = Image.open(io.BytesIO(png_bytes))
            data = pytesseract.image_to_data(image, output_type=Output.DICT)

            img_width, img_height = image.size
            scale_x = pdf_width / img_width if img_width else 1.0
            scale_y = pdf_height / img_height if img_height else 1.0

            # Group words into blocks/paragraphs
            lines_dict: dict[tuple[int, int], dict[str, Any]] = {}
            n_boxes = len(data["text"])
            total_conf = 0.0
            conf_count = 0

            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])
                if not text or conf < 0:
                    continue

                total_conf += conf
                conf_count += 1

                block_num = data["block_num"][i]
                line_num = data["line_num"][i]
                line_key = (block_num, line_num)

                x = float(data["left"][i]) * scale_x
                y = float(data["top"][i]) * scale_y
                w = float(data["width"][i]) * scale_x
                h = float(data["height"][i]) * scale_y

                if line_key not in lines_dict:
                    lines_dict[line_key] = {
                        "words": [text],
                        "bbox": {"x0": x, "y0": y, "x1": x + w, "y1": y + h},
                    }
                else:
                    lines_dict[line_key]["words"].append(text)
                    b = lines_dict[line_key]["bbox"]
                    b["x0"] = min(b["x0"], x)
                    b["y0"] = min(b["y0"], y)
                    b["x1"] = max(b["x1"], x + w)
                    b["y1"] = max(b["y1"], y + h)

            if not lines_dict:
                return []

            avg_confidence = (total_conf / conf_count / 100.0) if conf_count else 0.85
            elements = []

            for line_data in lines_dict.values():
                content = " ".join(line_data["words"])
                elements.append(
                    ParsedElement(
                        page_number=page_number,
                        bbox=line_data["bbox"],
                        type=ElementType.TEXT,
                        content=content,
                        ocr_applied=True,
                        ocr_confidence=avg_confidence,
                    )
                )

            return elements

        except Exception as e:
            logger.info(f"Primary Tesseract OCR not available: {e}")
            return []

    async def _run_llm_ocr_fallback(self, png_bytes: bytes, page_number: int) -> list[ParsedElement]:
        """LLM OCR fallback with bbox=None, ocr_applied=True, ocr_confidence=0.30 (low confidence marker)."""
        prompt = (
            "Transcribe all readable text from this document page image exactly as printed. "
            "Do not summarize or skip content."
        )
        image_part = ImagePart(bytes=png_bytes, mime_type="image/png")
        try:
            res = await self.llm_generator.generate(prompt=prompt, images=[image_part])
            extracted_text = res.text.strip()
            if not extracted_text:
                return []

            return [
                ParsedElement(
                    page_number=page_number,
                    bbox={"x0": 0.0, "y0": 0.0, "x1": 500.0, "y1": 700.0},  # Page-level fallback
                    type=ElementType.TEXT,
                    content=f"<ocr_fallback_page_{page_number}>\n{extracted_text}\n</ocr_fallback_page_{page_number}>",
                    ocr_applied=True,
                    ocr_confidence=0.30,  # Flagged low-confidence marker
                )
            ]
        except Exception as e:
            logger.error(f"LLM OCR fallback failed on page {page_number}: {e}")
            return []
