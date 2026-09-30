from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.db import init_db
from app.core.errors import RagnorkException, ragnork_exception_handler
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager initializing resources on startup."""
    logger.info("Initializing database schema...")
    try:
        await init_db()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.warning(f"Database initialization deferred or failed: {e}")
    yield
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Exception Handlers
app.add_exception_handler(RagnorkException, ragnork_exception_handler)

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {
        "app": settings.PROJECT_NAME,
        "status": "running",
        "docs_url": f"{settings.API_V1_STR}/docs",
    }


@app.get("/docs", include_in_schema=False)
async def redirect_docs():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"{settings.API_V1_STR}/docs")
