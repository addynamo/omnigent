"""Add reuse_session to scheduled_tasks.

Revision ID: zb2c3d4e5f6a
Revises: za2b3c4d5e6f
Create Date: 2026-08-21 00:00:00.000000

Adds a non-nullable ``reuse_session`` boolean column to ``scheduled_tasks``,
server-defaulted true. When true, a firing with a live
``last_run_conversation_id`` reuses that conversation (relaunching its runner
if needed) instead of always creating a new one — existing rows default to
true so they pick up the leak fix automatically; false restores the original
always-new-session behavior per task.

Additive. No backfill needed beyond the server default.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "zb2c3d4e5f6a"
down_revision: str | None = "za2b3c4d5e6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ``reuse_session`` to ``scheduled_tasks``, defaulting true."""
    op.add_column(
        "scheduled_tasks",
        sa.Column("reuse_session", sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade() -> None:
    """Remove ``reuse_session`` from ``scheduled_tasks``."""
    with op.batch_alter_table("scheduled_tasks") as batch_op:
        batch_op.drop_column("reuse_session")
