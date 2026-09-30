from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from app.core.config import settings
from app.core.logging import logger


class LLMResponse(BaseModel):
    text: str
    model_name: str = "gemini-2.5-flash"
    prompt_version: str = "qa_v1"


class LLMGenerator(ABC):
    """Abstract LLM Generator interface for grounded Q&A with citations."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response text based on prompt."""
        pass


class GeminiLLMGenerator(LLMGenerator):
    """Real Google Gemini API Generator implementation using google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self._client = None

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Gemini Client with model '{self.model_name}'")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Client: {e}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        if not self._client:
            logger.warning("No Gemini API key provided. Using Fake LLM Generator fallback.")
            fake = FakeLLMGenerator()
            return await fake.generate(prompt, system_prompt)

        try:
            # Call Google Gemini API
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            generated_text = response.text or ""
            return LLMResponse(
                text=generated_text,
                model_name=self.model_name,
                prompt_version="qa_v1",
            )
        except Exception as e:
            logger.error(f"Gemini API generation failed: {e}")
            fake = FakeLLMGenerator(
                response_text=f"API Error ({e}). Based on provided sources [1], please verify settings."
            )
            return await fake.generate(prompt, system_prompt)


class FakeLLMGenerator(LLMGenerator):
    """Fake LLM Generator for deterministic unit testing."""

    def __init__(self, response_text: Optional[str] = None):
        self.default_response = response_text or (
            "Based on the provided sources [1], the system architecture relies on PostgreSQL "
            "and Qdrant for hybrid search [2]."
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        return LLMResponse(
            text=self.default_response,
            model_name="fake-claude-3-5-sonnet",
            prompt_version="qa_v1",
        )


def get_llm_generator() -> LLMGenerator:
    """Factory function returning the configured LLM Generator."""
    if settings.GEMINI_API_KEY:
        return GeminiLLMGenerator()
    return FakeLLMGenerator()
