from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint verifying application & database connectivity."""
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {e!s}"

    return {
        "status": "ok",
        "database": db_status,
    }
