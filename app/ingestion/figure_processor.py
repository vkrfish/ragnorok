import asyncio
import hashlib
import io
import json
from typing import Any

from PIL import Image

from app.adapters.base import StorageAdapter
from app.adapters.llm import LLMGenerator, get_llm_generator
from app.core.config import settings
from app.core.logging import logger
from app.domain.schemas import ImagePart


class FigureProcessor:
    """Processor for figure crop storage, SHA256 caching, Vision LLM captioning, and structured chart extraction."""

    def __init__(self, storage: StorageAdapter, llm_generator: LLMGenerator | None = None):
        self.storage = storage
        self.llm_generator = llm_generator or get_llm_generator()
        self._hash_cache: dict[str, tuple[str, dict[str, Any] | None]] = {}

    def crop_figure_region(
        self,
        pdf_bytes: bytes,
        page_number: int,
        bbox: dict[str, float],
        max_dim: int = 1024,
    ) -> tuple[bytes, str]:
        """Render figure bbox region to downscaled PNG (~150 DPI) and compute SHA256 hash."""
        try:
            import fitz  # type: ignore[import-untyped]
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc[page_number - 1]

            rect = fitz.Rect(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
            # Clip rectangle to page bounds
            rect = rect.intersect(page.rect)
            if rect.is_empty:
                rect = page.rect

            pix = page.get_pixmap(dpi=settings.VISION_DPI, clip=rect)
            png_bytes = pix.tobytes("png")

            # Downscale using Pillow if larger than max_dim
            img = Image.open(io.BytesIO(png_bytes))
            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                png_bytes = buf.getvalue()

            image_hash = hashlib.sha256(png_bytes).hexdigest()
            return png_bytes, image_hash

        except Exception as e:
            logger.error(f"Failed to crop figure region: {e}")
            # Fallback 1x1 pixel image
            fallback_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xafA4\x00\x00\x00\x00IEND\xaeB`\x82"
            return fallback_bytes, hashlib.sha256(fallback_bytes).hexdigest()

    async def save_crop_image(
        self,
        notebook_id: str,
        source_id: str,
        element_id: str,
        crop_bytes: bytes,
    ) -> str:
        """Save figure crop through the StorageAdapter interface."""
        crop_filename = f"{element_id}.png"
        crop_file_obj = io.BytesIO(crop_bytes)
        stored_path = await self.storage.save(
            file_obj=crop_file_obj,
            notebook_id=notebook_id,
            source_id=source_id,
            filename=f"crops/{crop_filename}",
            content_type="image/png",
        )
        return stored_path

    async def process_figure_caption(
        self,
        crop_bytes: bytes,
        image_hash: str,
        pdf_caption: str = "",
        max_retries: int = 2,
    ) -> tuple[str, dict[str, Any] | None, str, str]:
        """Caption figure crop using Vision LLM with SHA256 caching and retry backoff.
        Returns: (caption_text, structured_chart_json, model_name, prompt_version)
        """
        # 1. Check SHA256 Hash Cache
        if image_hash in self._hash_cache:
            logger.info(f"SHA256 cache hit for figure image hash {image_hash[:10]}...")
            cached_caption, cached_json = self._hash_cache[image_hash]
            return cached_caption, cached_json, settings.GEMINI_MODEL, "caption_v1"

        if not settings.ENABLE_VISION:
            return f"Figure image (vision disabled in dev mode). {pdf_caption}".strip(), None, "dev-disabled", "caption_v1"

        image_part = ImagePart(bytes=crop_bytes, mime_type="image/png")

        # 2. Vision Captioning with Retry Backoff
        caption_prompt = (
            f"Provide a detailed, search-optimized textual summary describing key figures, data points, "
            f"and trends shown in this image crop. "
            f"Original PDF caption text: '{pdf_caption}'."
        )

        caption_text = ""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                res = await self.llm_generator.generate(prompt=caption_prompt, images=[image_part])
                caption_text = res.text.strip()
                if caption_text:
                    break
            except Exception as e:
                last_exception = e
                logger.warning(f"Figure caption attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (2 ** attempt))

        if not caption_text:
            logger.error(f"Figure caption failed after retries: {last_exception}")
            caption_text = f"Figure region. {pdf_caption}".strip()

        # 3. Structured Chart Extraction
        chart_json = await self._extract_structured_chart_json(image_part)

        # Update SHA256 Cache
        self._hash_cache[image_hash] = (caption_text, chart_json)

        return caption_text, chart_json, settings.GEMINI_MODEL, "caption_v1"

    async def _extract_structured_chart_json(self, image_part: ImagePart) -> dict[str, Any] | None:
        """Extract structured JSON for charts if applicable."""
        chart_prompt = (
            "Extract structured chart data into JSON format with keys: chart_type, title, axes, series. "
            "Set `is_label_read: true` for explicit text labels and `false` for estimated values. "
            "Output valid JSON only."
        )
        try:
            res = await self.llm_generator.generate(prompt=chart_prompt, images=[image_part])
            text = res.text.strip()
            # Clean json fences if present
            text = text.removeprefix("```json")
            text = text.removesuffix("```")
            text = text.strip()
            return json.loads(text)
        except Exception:
            return None
