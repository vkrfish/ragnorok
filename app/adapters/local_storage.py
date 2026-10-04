import os
import shutil
from pathlib import Path
from typing import BinaryIO

from app.adapters.base import StorageAdapter
from app.core.errors import StorageError
from app.core.logging import logger


class LocalStorageAdapter(StorageAdapter):
    """Local filesystem implementation of StorageAdapter."""

    def __init__(self, base_dir: str = "./media"):
        self.base_path = Path(base_dir).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save(
        self,
        file_obj: BinaryIO,
        notebook_id: str,
        source_id: str,
        filename: str,
        content_type: str = "application/pdf",
    ) -> str:
        try:
            target_dir = self.base_path / notebook_id / source_id
            target_file = target_dir / filename
            target_file.parent.mkdir(parents=True, exist_ok=True)

            file_obj.seek(0)
            with open(target_file, "wb") as buffer:
                shutil.copyfileobj(file_obj, buffer)

            relative_path = str(target_file.relative_to(self.base_path))
            logger.info(f"Saved file to local storage: {relative_path}")
            return relative_path
        except Exception as e:
            logger.error(f"Failed to save file to local storage: {e!s}")
            raise StorageError(f"Local storage save failed: {e!s}")

    async def get(self, file_path: str) -> bytes:
        try:
            full_path = self.base_path / file_path
            if not full_path.exists():
                raise StorageError(f"File not found: {file_path}")
            with open(full_path, "rb") as f:
                return f.read()
        except Exception as e:
            raise StorageError(f"Local storage get failed: {e!s}")

    async def delete(self, file_path: str) -> bool:
        try:
            full_path = self.base_path / file_path
            if full_path.exists():
                os.remove(full_path)
                return True
            return False
        except Exception as e:
            raise StorageError(f"Local storage delete failed: {e!s}")

    async def exists(self, file_path: str) -> bool:
        full_path = self.base_path / file_path
        return full_path.exists()
