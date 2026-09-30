from app.adapters.base import StorageAdapter
from app.adapters.local_storage import LocalStorageAdapter
from app.adapters.supabase_storage import SupabaseStorageAdapter
from app.core.config import settings


def get_storage_adapter() -> StorageAdapter:
    """Factory function returning the configured storage adapter implementation."""
    if settings.STORAGE_BACKEND == "supabase":
        return SupabaseStorageAdapter()
    return LocalStorageAdapter(base_dir=settings.LOCAL_STORAGE_DIR)
