import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_openapi_schema_snapshot(client: AsyncClient):
    res = await client.get("/v1/openapi.json")
    assert res.status_code == 200
    schema = res.json()

    assert schema["openapi"].startswith("3.")
    assert "paths" in schema
    paths = schema["paths"]

    # Verify /v1 routes are present in OpenAPI schema
    assert "/v1/notebooks" in paths
    assert "/v1/notebooks/{notebook_id}/chat" in paths
    assert "/v1/notebooks/{notebook_id}/chat/stream" in paths
    assert "/v1/notebooks/{notebook_id}/sources/{source_id}" in paths
    assert "/v1/sources/{source_id}/pages/{page_number}" in paths
    assert "/v1/sources/{source_id}/pages/{page_number}/image" in paths
    assert "/v1/sources/{source_id}/elements/{element_id}" in paths
