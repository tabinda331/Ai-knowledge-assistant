# Database and local PostgreSQL

The backend uses ordinary PostgreSQL connectivity through SQLAlchemy and psycopg. It has no Supabase SDK dependency. PostgreSQL stores document metadata, chunks, embeddings, and conversation history; PDF files remain in local document storage.

## Current schema

- documents: source metadata, lifecycle status, JSONB metadata, and timestamps.
- document_chunks: ordered text passages, JSONB page metadata, a foreign key to documents, and nullable vector(1024) embeddings. (document_id, chunk_index) is unique; deleting a document cascades to its chunks.
- conversations: conversation identity and timestamps.
- messages: ordered user/assistant turns, source JSON, and metadata. (conversation_id, message_index) is unique.

The initial migration enables the vector extension and creates the document tables. Later revisions add conversation/message tables and deterministic message ordering. The current Alembic head is 0003_message_order.

## Configuration and migration workflow

backend/app/config.py loads backend/.env and reads DATABASE_URL from the process environment. A process-level value takes precedence over the .env value. backend/app/db/session.py creates the SQLAlchemy engine lazily and caches it per URL.

Alembic reads the same DATABASE_URL through backend/alembic/env.py; the URL field in backend/alembic.ini is intentionally blank. Run migrations explicitly from backend/:

~~~powershell
python -m alembic upgrade head
python -m alembic current
~~~

FastAPI startup does not create tables or migrate the database. The initial migration requires permission to create the pgvector extension.

## Disposable local database

For a local PostgreSQL 17/pgvector container, use the pgvector/pgvector:pg17 image and bind its port to 127.0.0.1, for example 127.0.0.1:55432:5432. See the repository Windows setup for a complete Docker and PowerShell workflow.

Set a process-scoped local DATABASE_URL before migrations or manual backend requests when an existing backend/.env could point to another database. A local URL has this shape:

~~~text
postgresql+psycopg://ragtest:REPLACE_WITH_A_LOCAL_TEST_PASSWORD@127.0.0.1:55432/ragtest
~~~

The password is a placeholder. Never copy real database credentials into documentation, frontend configuration, or version control. The browser calls FastAPI and never connects directly to PostgreSQL.

## Verification

The automated backend tests use fakes and in-memory SQLite; they do not require PostgreSQL. A fresh local database can be checked with python -m alembic current and the backend GET /health/db endpoint.

backend/scripts/verify_database.py performs additional corpus checks and by default expects a specific seeded document with 136 embedded chunks. It is not intended to pass on a fresh database without that document.

Local PDFs are stored under backend/storage/ by default. Set DOCUMENT_STORAGE_DIR to a separate local path to keep test uploads outside that default. This directory is ignored by Git.
