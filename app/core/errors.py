from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class APIErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict[str, Any] | None = None


class RagnorkException(Exception):
    """Base exception for Ragnork domain errors."""
    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class NotFoundError(RagnorkException):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class ForbiddenError(RagnorkException):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="FORBIDDEN",
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
        )


class StorageError(RagnorkException):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class DuplicateResourceError(RagnorkException):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="DUPLICATE_RESOURCE",
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


async def ragnork_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, RagnorkException):
        status_code = exc.status_code
        error_code = exc.error_code
        message = exc.message
        details = exc.details
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_code = "INTERNAL_ERROR"
        message = str(exc)
        details = {}

    return JSONResponse(
        status_code=status_code,
        content=APIErrorResponse(
            error_code=error_code,
            message=message,
            details=details,
        ).model_dump(),
    )
