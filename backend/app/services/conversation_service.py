from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_conversation_history_messages
from app.models.conversation import Conversation, Message

_REFERENCE = re.compile(r"\b(it|they|them|this|that|these|those|its|their)\b", re.IGNORECASE)
_PROPER_NOUN = re.compile(r"\b(?:[A-Z][A-Za-z0-9-]*)(?:\s+[A-Z0-9][A-Za-z0-9-]*){0,4}\b")
_NON_ENTITIES = {"a", "an", "are", "can", "does", "how", "is", "tell", "the", "this", "what", "when", "where", "which", "who", "why"}


def create_conversation(db: Session) -> Conversation:
    conversation = Conversation(id=uuid.uuid4())
    db.add(conversation)
    return conversation


def get_conversation(db: Session, conversation_id: uuid.UUID) -> Conversation | None:
    return db.get(Conversation, conversation_id)


def load_history(db: Session, conversation_id: uuid.UUID) -> list[Message]:
    limit = get_conversation_history_messages()
    if limit == 0:
        return []
    newest = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.message_index.desc())
        .limit(limit)
        .subquery()
    )
    return list(
        db.scalars(
            select(Message)
            .join(newest, Message.id == newest.c.id)
            .order_by(Message.message_index)
        ).all()
    )


def resolve_retrieval_query(question: str, history: list[Message]) -> str:
    """Resolve simple references from prior user wording, never assistant claims."""
    if not history or not _REFERENCE.search(question):
        return question
    prior_user_questions = [message.content for message in history if message.role == "user"]
    subject = None
    for prior in reversed(prior_user_questions):
        candidates = [
            candidate for candidate in _PROPER_NOUN.findall(prior)
            if candidate.split()[0].casefold() not in _NON_ENTITIES
        ]
        if candidates:
            subject = candidates[-1].strip()
            break
    if not subject:
        return question
    return _REFERENCE.sub(subject, question)


def persist_turn(
    db: Session,
    conversation: Conversation,
    question: str,
    answer: str,
    sources: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    # Serialize turns per conversation so concurrent requests cannot allocate
    # overlapping message indexes. New conversations are flushed before locking.
    db.flush()
    locked_conversation = db.execute(
        select(Conversation)
        .where(Conversation.id == conversation.id)
        .with_for_update()
    ).scalar_one()
    current_max = db.scalar(
        select(func.coalesce(func.max(Message.message_index), 0)).where(
            Message.conversation_id == conversation.id
        )
    )
    first_index = int(current_max or 0) + 1
    db.add_all(
        [
            Message(
                conversation_id=conversation.id,
                message_index=first_index,
                role="user",
                content=question,
            ),
            Message(
                conversation_id=conversation.id,
                message_index=first_index + 1,
                role="assistant",
                content=answer,
                sources=sources,
                metadata_json=metadata,
            ),
        ]
    )
    locked_conversation.updated_at = datetime.now(timezone.utc)
    db.add(locked_conversation)
    db.commit()