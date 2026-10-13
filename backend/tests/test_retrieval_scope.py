from __future__ import annotations

import json
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.services.bm25_retrieval import _load_chunks
from app.services.retrieval_scope import ready_document_ids_statement


def test_global_readiness_requires_completed_status_and_embedding_for_every_chunk() -> None:
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE documents (id CHAR(32) PRIMARY KEY, status VARCHAR(20))"))
        connection.execute(text(
            "CREATE TABLE document_chunks (id CHAR(32) PRIMARY KEY, document_id CHAR(32), "
            "chunk_index INTEGER, content TEXT, metadata TEXT, embedding TEXT)"
        ))

        states = {
            uuid.uuid4(): ("completed", ["vector", "vector"]),
            uuid.uuid4(): ("completed", ["vector", None]),
            uuid.uuid4(): ("completed", [None]),
            uuid.uuid4(): ("completed", []),
            uuid.uuid4(): ("failed", ["vector"]),
            uuid.uuid4(): ("processing", ["vector"]),
        }
        ready_id = next(iter(states))
        for document_id, (status, embeddings) in states.items():
            connection.execute(
                text("INSERT INTO documents (id, status) VALUES (:id, :status)"),
                {"id": document_id.hex, "status": status},
            )
            for index, embedding in enumerate(embeddings):
                connection.execute(
                    text(
                        "INSERT INTO document_chunks "
                        "(id, document_id, chunk_index, content, metadata, embedding) "
                        "VALUES (:id, :document_id, :chunk_index, :content, :metadata, :embedding)"
                    ),
                    {
                        "id": uuid.uuid4().hex,
                        "document_id": document_id.hex,
                        "chunk_index": index,
                        "content": f"ready evidence token {index}" if document_id == ready_id else "nonready evidence token",
                        "metadata": json.dumps({"page_start": index + 1}),
                        "embedding": embedding,
                    },
                )

    with Session(engine) as session:
        eligible_ids = set(session.scalars(ready_document_ids_statement()).all())
        chunks = _load_chunks(session, None)

    assert eligible_ids == {ready_id}
    assert chunks and {chunk.document_id for chunk in chunks} == {ready_id}
