from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from app.models.conversation import Conversation, Message
from app.services import conversation_service


class _ScalarResult:
    def __init__(self, value: Conversation):
        self._value = value

    def scalar_one(self) -> Conversation:
        return self._value


class _ScalarsResult:
    def __init__(self, values: list[Message]):
        self._values = values

    def all(self) -> list[Message]:
        return self._values


class FakeConversationSession:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.pending: list[Message] = []
        self.conversation: Conversation | None = None
        self.fixed_timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.history_limit = 10

    def add(self, item: Any) -> None:
        if isinstance(item, Conversation):
            self.conversation = item

    def add_all(self, items: list[Message]) -> None:
        self.pending.extend(items)
        for item in items:
            item.created_at = self.fixed_timestamp

    def flush(self) -> None:
        assert self.conversation is not None

    def execute(self, statement: Any) -> _ScalarResult:
        sql = str(statement.compile(dialect=postgresql.dialect())).upper()
        assert "FOR UPDATE" in sql
        assert self.conversation is not None
        return _ScalarResult(self.conversation)

    def scalar(self, statement: Any) -> int:
        sql = str(statement.compile(dialect=postgresql.dialect())).lower()
        assert "max(messages.message_index)" in sql
        return max((message.message_index for message in self.messages), default=0)

    def commit(self) -> None:
        self.messages.extend(self.pending)
        self.pending.clear()

    def scalars(self, statement: Any) -> _ScalarsResult:
        sql = str(statement.compile(dialect=postgresql.dialect())).lower()
        assert "order by messages.message_index desc" in sql
        assert "order by messages.message_index" in sql
        # Mimic the rows selected by the bounded subquery; ordering is checked
        # above against the SQL emitted by the history loader.
        values = sorted(self.messages, key=lambda message: message.message_index)[
            -self.history_limit:
        ]
        return _ScalarsResult(values)


def save_turn(session: FakeConversationSession, question: str, answer: str) -> None:
    assert session.conversation is not None
    conversation_service.persist_turn(
        session,
        session.conversation,
        question,
        answer,
        sources=[],
        metadata={"grounded": True},
    )


def test_message_indexes_preserve_turn_order_even_with_equal_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conversation_service, "get_conversation_history_messages", lambda: 10)
    session = FakeConversationSession()
    session.conversation = conversation_service.create_conversation(session)

    save_turn(session, "Question one", "Answer one")
    save_turn(session, "Question two", "Answer two")

    assert [message.message_index for message in session.messages] == [1, 2, 3, 4]
    assert [message.role for message in session.messages] == [
        "user", "assistant", "user", "assistant"
    ]
    assert len({message.created_at for message in session.messages}) == 1

    history = conversation_service.load_history(session, session.conversation.id)
    assert [message.role for message in history] == [
        "user", "assistant", "user", "assistant"
    ]
    assert [message.content for message in history] == [
        "Question one", "Answer one", "Question two", "Answer two"
    ]


def test_history_limit_keeps_the_newest_messages_in_sequence_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conversation_service, "get_conversation_history_messages", lambda: 2)
    session = FakeConversationSession()
    session.history_limit = 2
    session.conversation = conversation_service.create_conversation(session)
    save_turn(session, "Question one", "Answer one")
    save_turn(session, "Question two", "Answer two")

    history = conversation_service.load_history(session, session.conversation.id)

    assert [message.content for message in history] == ["Question two", "Answer two"]
    assert [message.message_index for message in history] == [3, 4]
