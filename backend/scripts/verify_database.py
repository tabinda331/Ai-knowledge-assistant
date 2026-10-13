from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import get_engine

TARGET_DOCUMENT_ID = "a33ebd66-bb12-4314-ae89-7a0549fb3868"


def get_expected_migration_heads() -> set[str]:
    alembic_ini = Path(__file__).resolve().parents[1] / "alembic.ini"
    return set(ScriptDirectory.from_config(Config(str(alembic_ini))).get_heads())


def migration_heads_are_applied(
    applied_revisions: Iterable[str], expected_heads: set[str]
) -> bool:
    return bool(expected_heads) and expected_heads.issubset(set(applied_revisions))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the database schema and embedded document.")
    parser.add_argument("--document-id", type=uuid.UUID, default=TARGET_DOCUMENT_ID)
    parser.add_argument("--expected-chunks", type=int, default=136)
    arguments = parser.parse_args()
    expected_heads = get_expected_migration_heads()

    engine = get_engine()
    if engine is None:
        print("Database verification failed: DATABASE_URL is not configured.")
        return 1

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            extension_available = connection.scalar(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )
            inspector = inspect(connection)
            tables = set(inspector.get_table_names())
            embedding_schema = None
            if "document_chunks" in tables:
                embedding_schema = connection.execute(
                    text(
                        """
                        SELECT format_type(attribute.atttypid, attribute.atttypmod) AS data_type,
                               NOT attribute.attnotnull AS is_nullable
                        FROM pg_attribute AS attribute
                        JOIN pg_class AS relation ON relation.oid = attribute.attrelid
                        JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                        WHERE namespace.nspname = current_schema()
                          AND relation.relname = 'document_chunks'
                          AND attribute.attname = 'embedding'
                          AND attribute.attnum > 0
                          AND NOT attribute.attisdropped
                        """
                    )
                ).one_or_none()
            applied_revisions: set[str] = set()
            if "alembic_version" in tables:
                applied_revisions = set(
                    connection.scalars(text("SELECT version_num FROM alembic_version")).all()
                )
            migration_applied = migration_heads_are_applied(applied_revisions, expected_heads)
            embedding_stats = None
            target_document = connection.execute(
                text(
                    """
                    SELECT document.title,
                           COUNT(chunk.id) AS total_chunks,
                           COUNT(chunk.embedding) AS embedded_chunks,
                           COUNT(DISTINCT chunk.chunk_index) AS unique_chunk_indexes,
                           MAX(vector_dims(chunk.embedding)) AS embedding_dimension
                    FROM documents AS document
                    LEFT JOIN document_chunks AS chunk ON chunk.document_id = document.id
                    WHERE document.id = :document_id
                    GROUP BY document.id, document.title
                    """
                ),
                {"document_id": arguments.document_id},
            ).mappings().one_or_none()
            if target_document is not None:
                embedding_stats = dict(target_document)
    except SQLAlchemyError:
        print("Database verification failed: unable to inspect database schema.")
        return 1

    checks = {
        "PostgreSQL connection": True,
        "pgvector extension": bool(extension_available),
        "documents table": "documents" in tables,
        "document_chunks table": "document_chunks" in tables,
        "document_chunks.embedding column": embedding_schema is not None,
        "embedding vector(1024)": bool(
            embedding_schema and embedding_schema.data_type.lower() == "vector(1024)"
        ),
        "embedding nullable": bool(embedding_schema and embedding_schema.is_nullable),
        "Alembic migration recorded": migration_applied,
        "target document exists": embedding_stats is not None,
        "target chunk count matches": bool(
            embedding_stats and embedding_stats["total_chunks"] == arguments.expected_chunks
        ),
        "all target chunks embedded": bool(
            embedding_stats
            and embedding_stats["embedded_chunks"] == embedding_stats["total_chunks"]
        ),
        "target vectors are 1024-dimensional": bool(
            embedding_stats
            and embedding_stats["embedding_dimension"] == 1024
            and embedding_stats["embedded_chunks"] == embedding_stats["total_chunks"]
        ),
        "target chunk indexes unique": bool(
            embedding_stats
            and embedding_stats["unique_chunk_indexes"] == embedding_stats["total_chunks"]
        ),
    }
    for check, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {check}")
    if embedding_stats is not None:
        print(
            "Target document: "
            f"{embedding_stats['title']} | "
            f"chunks={embedding_stats['total_chunks']} | "
            f"embedded={embedding_stats['embedded_chunks']} | "
            f"dimension={embedding_stats['embedding_dimension']}"
        )

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
