from typing import BinaryIO
from supabase import create_client, Client
from app.adapters.base import StorageAdapter
from app.adapters.local_storage import LocalStorageAdapter
from app.core.config import settings
from app.core.errors import StorageError
from app.core.logging import logger


class SupabaseStorageAdapter(StorageAdapter):
    """Supabase Storage implementation with automatic local storage fallback for permission/signature errors."""

    def __init__(self):
        self.bucket_name = settings.SUPABASE_STORAGE_BUCKET
        self._local_fallback = LocalStorageAdapter(base_dir=settings.LOCAL_STORAGE_DIR)

    @property
    def client(self) -> Client:
        return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    async def save(
        self,
        file_obj: BinaryIO,
        notebook_id: str,
        source_id: str,
        filename: str,
        content_type: str = "application/pdf",
    ) -> str:
        storage_path = f"{notebook_id}/{source_id}/{filename}"
        try:
            file_obj.seek(0)
            file_bytes = file_obj.read()

            self.client.storage.from_(self.bucket_name).upload(
                file=file_bytes,
                path=storage_path,
                file_options={"content-type": content_type, "x-upsert": "true"},
            )
            logger.info(f"Saved file to Supabase storage bucket '{self.bucket_name}': {storage_path}")
            return storage_path
        except Exception as e:
            err_str = str(e)
            logger.warning(
                f"Supabase storage save failed ({err_str}). Falling back to local storage."
            )
            return await self._local_fallback.save(
                file_obj=file_obj,
                notebook_id=notebook_id,
                source_id=source_id,
                filename=filename,
                content_type=content_type,
            )

    async def get(self, file_path: str) -> bytes:
        try:
            return self.client.storage.from_(self.bucket_name).download(file_path)
        except Exception as e:
            if await self._local_fallback.exists(file_path):
                return await self._local_fallback.get(file_path)
            raise StorageError(f"Storage get failed: {str(e)}")

    async def delete(self, file_path: str) -> bool:
        try:
            self.client.storage.from_(self.bucket_name).remove([file_path])
            return True
        except Exception:
            return await self._local_fallback.delete(file_path)

    async def exists(self, file_path: str) -> bool:
        try:
            folder = "/".join(file_path.split("/")[:-1])
            filename = file_path.split("/")[-1]
            res = self.client.storage.from_(self.bucket_name).list(path=folder)
            if any(item.get("name") == filename for item in res):
                return True
        except Exception:
            pass
        return await self._local_fallback.exists(file_path)
