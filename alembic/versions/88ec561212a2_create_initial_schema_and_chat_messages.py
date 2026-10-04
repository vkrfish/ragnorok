"""create_initial_schema_and_chat_messages

Revision ID: 88ec561212a2
Revises: 
Create Date: 2026-10-03 22:49:53.873076

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '88ec561212a2'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. notebooks
    op.create_table(
        'notebooks',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=255), nullable=False, index=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        if_not_exists=True,
    )

    # 2. sources
    op.create_table(
        'sources',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('notebook_id', sa.String(length=36), sa.ForeignKey('notebooks.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.String(length=255), nullable=False, index=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False, index=True),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('mime_type', sa.String(length=100), server_default='application/pdf'),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, index=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('figures_pending', sa.Integer(), server_default='0', nullable=False),
        sa.Column('figures_completed', sa.Integer(), server_default='0', nullable=False),
        sa.Column('figures_failed', sa.Integer(), server_default='0', nullable=False),
        sa.Column('ocr_pages_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('ingestion_cost_usd', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        if_not_exists=True,
    )

    # 3. elements
    op.create_table(
        'elements',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('source_id', sa.String(length=36), sa.ForeignKey('sources.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('bbox', sa.JSON(), nullable=True),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('image_path', sa.Text(), nullable=True),
        sa.Column('image_hash', sa.String(length=64), nullable=True),
        sa.Column('caption_status', sa.String(length=50), server_default='pending'),
        sa.Column('structured_chart_json', sa.JSON(), nullable=True),
        sa.Column('ocr_applied', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('ocr_confidence', sa.Float(), nullable=True),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('prompt_version', sa.String(length=50), nullable=True),
        sa.Column('parser_version', sa.String(length=50), server_default='1.0.0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        if_not_exists=True,
    )

    # 4. chunks
    op.create_table(
        'chunks',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('source_id', sa.String(length=36), sa.ForeignKey('sources.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('notebook_id', sa.String(length=36), nullable=False, index=True),
        sa.Column('user_id', sa.String(length=255), nullable=False, index=True),
        sa.Column('element_id', sa.String(length=36), sa.ForeignKey('elements.id', ondelete='SET NULL'), nullable=True),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embeddable_text', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False),
        sa.Column('heading_path', sa.Text(), nullable=True),
        sa.Column('bbox', sa.JSON(), nullable=True),
        sa.Column('embedding_model_version', sa.String(length=50), server_default='bge-m3-v1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        if_not_exists=True,
    )

    # 5. chat_messages
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('conversation_id', sa.String(length=36), nullable=False, index=True),
        sa.Column('notebook_id', sa.String(length=36), sa.ForeignKey('notebooks.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.String(length=255), nullable=False, index=True),
        sa.Column('role', sa.String(length=20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('citations', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_table('chat_messages', if_exists=True)
    op.drop_table('chunks', if_exists=True)
    op.drop_table('elements', if_exists=True)
    op.drop_table('sources', if_exists=True)
    op.drop_table('notebooks', if_exists=True)
