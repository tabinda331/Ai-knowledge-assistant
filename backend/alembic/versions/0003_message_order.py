"""Add deterministic per-conversation message order."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003_message_order"
down_revision: Union[str, None] = "0002_conversation_memory"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("message_index", sa.Integer(), nullable=True))
    op.execute(
        sa.text(
            "WITH ordered_messages AS ("
            " SELECT id, ROW_NUMBER() OVER ("
            "   PARTITION BY conversation_id ORDER BY created_at, id"
            " ) AS assigned_index FROM messages"
            ") UPDATE messages AS message"
            " SET message_index = ordered_messages.assigned_index"
            " FROM ordered_messages WHERE message.id = ordered_messages.id"
        )
    )
    op.alter_column("messages", "message_index", nullable=False)
    op.drop_index("ix_messages_conversation_created_id", table_name="messages")
    op.create_unique_constraint(
        "uq_messages_conversation_message_index",
        "messages",
        ["conversation_id", "message_index"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_messages_conversation_message_index", "messages", type_="unique"
    )
    op.drop_column("messages", "message_index")
    op.create_index(
        "ix_messages_conversation_created_id",
        "messages",
        ["conversation_id", "created_at", "id"],
        unique=False,
    )
