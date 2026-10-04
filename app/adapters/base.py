from abc import ABC, abstractmethod
from typing import BinaryIO


class StorageAdapter(ABC):
    """Abstract interface for binary storage ports (Local File System, Supabase Storage, S3)."""

    @abstractmethod
    async def save(
        self,
        file_obj: BinaryIO,
        notebook_id: str,
        source_id: str,
        filename: str,
        content_type: str = "application/pdf",
    ) -> str:
        """Save file data to storage. Returns the stored file path/identifier."""

    @abstractmethod
    async def get(self, file_path: str) -> bytes:
        """Retrieve file content as bytes."""

    @abstractmethod
    async def delete(self, file_path: str) -> bool:
        """Delete a file from storage by path."""

    @abstractmethod
    async def exists(self, file_path: str) -> bool:
        """Check if a file exists at the given path."""
