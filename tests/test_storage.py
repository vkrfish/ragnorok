import io
import tempfile
import pytest
from app.adapters.local_storage import LocalStorageAdapter


@pytest.mark.asyncio
async def test_local_storage_adapter():
    with tempfile.TemporaryDirectory() as temp_dir:
        adapter = LocalStorageAdapter(base_dir=temp_dir)
        content = b"%PDF-1.4 Mock PDF Content"
        file_obj = io.BytesIO(content)

        # Save
        rel_path = await adapter.save(
            file_obj=file_obj,
            notebook_id="nb-123",
            source_id="src-456",
            filename="test.pdf",
        )
        assert "nb-123" in rel_path
        assert "test.pdf" in rel_path

        # Exists
        exists = await adapter.exists(rel_path)
        assert exists is True

        # Get
        retrieved_bytes = await adapter.get(rel_path)
        assert retrieved_bytes == content

        # Delete
        deleted = await adapter.delete(rel_path)
        assert deleted is True
        assert await adapter.exists(rel_path) is False
