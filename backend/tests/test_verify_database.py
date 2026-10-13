from __future__ import annotations

from scripts.verify_database import (
    get_expected_migration_heads,
    migration_heads_are_applied,
)


def test_database_verifier_accepts_current_alembic_head() -> None:
    expected_heads = get_expected_migration_heads()

    assert expected_heads == {"0003_message_order"}
    assert migration_heads_are_applied(expected_heads, expected_heads)


def test_database_verifier_rejects_a_superseded_migration_revision() -> None:
    expected_heads = get_expected_migration_heads()

    assert not migration_heads_are_applied({"0001_initial_database"}, expected_heads)
