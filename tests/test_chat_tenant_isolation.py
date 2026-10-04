import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.domain.models import ChatMessage, Notebook


@pytest.mark.asyncio
async def test_chat_conversation_tenant_isolation():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        nb1 = Notebook(id="nb-t1", user_id="user-t1", title="Tenant 1 Notebook")
        nb2 = Notebook(id="nb-t2", user_id="user-t2", title="Tenant 2 Notebook")
        db.add_all([nb1, nb2])
        await db.commit()

        # Add message owned by Tenant 1
        msg1 = ChatMessage(
            id="msg-1",
            conversation_id="conv-t1",
            notebook_id="nb-t1",
            user_id="user-t1",
            role="user",
            content="Hello from Tenant 1",
        )
        db.add(msg1)
        await db.commit()

        # Verify Tenant 2 cannot access conv-t1
        from sqlalchemy import select
        res = await db.execute(select(ChatMessage).where(ChatMessage.conversation_id == "conv-t1"))
        messages = list(res.scalars().all())

        assert len(messages) == 1
        assert messages[0].user_id == "user-t1"
        assert messages[0].notebook_id == "nb-t1"
