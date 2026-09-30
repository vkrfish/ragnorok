from typing import List, Optional
from fastapi import APIRouter, Depends, Header, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.db import get_db
from app.core.errors import NotFoundError
from app.domain.models import Notebook
from app.domain.schemas import NotebookCreate, NotebookResponse

router = APIRouter()


def get_user_id(x_user_id: Optional[str] = Header(None, alias="X-User-ID")) -> str:
    """Extract user_id for tenant isolation boundary."""
    return x_user_id or "default_user"


@router.post("/", response_model=NotebookResponse, status_code=status.HTTP_201_CREATED)
async def create_notebook(
    payload: NotebookCreate,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """Create a new notebook associated with user_id."""
    notebook = Notebook(
        user_id=user_id,
        title=payload.title,
    )
    db.add(notebook)
    await db.commit()
    await db.refresh(notebook)
    return notebook


@router.get("/", response_model=List[NotebookResponse])
async def list_notebooks(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """List all notebooks owned by user_id."""
    result = await db.execute(
        select(Notebook).where(Notebook.user_id == user_id)
    )
    notebooks = result.scalars().all()
    return notebooks


@router.get("/{notebook_id}", response_model=NotebookResponse)
async def get_notebook(
    notebook_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """Get a specific notebook by notebook_id and user_id."""
    result = await db.execute(
        select(Notebook).where(
            Notebook.id == notebook_id,
            Notebook.user_id == user_id,
        )
    )
    notebook = result.scalar_one_or_none()
    if not notebook:
        raise NotFoundError(f"Notebook '{notebook_id}' not found.")
    return notebook
