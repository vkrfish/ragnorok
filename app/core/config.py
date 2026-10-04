from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Notebook RAG"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"

    # Database Configuration
    DATABASE_URL: str = "sqlite+aiosqlite:///./ragnork.db"

    # LLM Settings
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"

    # Supabase Configuration
    SUPABASE_URL: str = "https://your-supabase-project.supabase.co"
    SUPABASE_KEY: str = "your-supabase-key"
    SUPABASE_STORAGE_BUCKET: str = "raw-sources"

    # Storage Backend
    STORAGE_BACKEND: Literal["local", "supabase"] = "local"
    LOCAL_STORAGE_DIR: str = "./media"

    # Redis & Workers
    REDIS_URL: str = "redis://localhost:6379/0"

    # Qdrant Vector DB
    QDRANT_URL: str = "http://localhost:6333"

    # Phase 3 Configuration
    PARSER_BACKEND: Literal["pymupdf", "docling"] = "pymupdf"
    OCR_ENGINE: Literal["tesseract", "llm_fallback"] = "tesseract"
    ENABLE_VISION: bool = True
    MAX_CAPTION_CONCURRENCY: int = 3
    MAX_FIGURES_PER_DOC: int = 20
    VISION_DPI: int = 150

    # Phase 4 Configuration
    RERANKER_MODEL: str = "bge-reranker-v2-m3"
    RERANK_REFUSAL_THRESHOLD: float = 0.35
    RRF_K: int = 60
    MAX_RERANK_CANDIDATES: int = 35
    MAX_PARENT_CONTEXT_TOKENS: int = 2000
    MAX_CHAT_HISTORY_TURNS: int = 6
    ENABLE_QUERY_REWRITE: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
