from fastapi import APIRouter

from app.api.v1 import chat, health, notebooks, sources

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(notebooks.router, prefix="/notebooks", tags=["Notebooks"])
api_router.include_router(sources.router, prefix="/notebooks", tags=["Sources"])
api_router.include_router(chat.router, prefix="/notebooks", tags=["Chat"])
