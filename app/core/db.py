from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Base ORM model class."""


from typing import Any

# Create async engine for PostgreSQL (Supabase / PgBouncer) or SQLite
connect_args: dict[str, Any] = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    # Disable prepared statement caching for PgBouncer compatibility on port 6543
    connect_args = {
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
    }

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
    connect_args=connect_args,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Initialize DB tables asynchronously (creates tables if not existing)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection yield for DB session management."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
