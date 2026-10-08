# Database Architecture

## Phase 2 Scope

PostgreSQL is the system of record for source documents and their future text chunks. pgvector provides a native PostgreSQL vector column so embeddings can eventually be stored and queried without exposing the database to the browser. Supabase is the planned host because managed PostgreSQL reduces database provisioning and maintenance work. The backend still uses ordinary PostgreSQL connectivity and has no dependency on a Supabase SDK, so it remains portable to other PostgreSQL hosts.

## Schema

- `documents` stores source identity, title, optional filename/URL/MIME type, lifecycle status, JSONB metadata, and timestamps. Status is constrained to `pending`, `processing`, `completed`, or `failed`.
- `document_chunks` stores ordered chunks with a foreign key to `documents`, content, JSONB metadata, a creation timestamp, and a unique `(document_id, chunk_index)` constraint. Deleting a document cascades to its chunks.
- `document_chunks.embedding` is nullable `vector(1024)`, matching the planned BAAI/bge-m3 embedding size. No embeddings are produced in this phase.

The unique B-tree index for `(document_id, chunk_index)` also supports lookups by its leading `document_id` key, so a redundant single-column index is not created. No vector similarity index is created yet: there are no real vectors, and a later retrieval phase should select an index method and parameters based on workload and measured query behavior.

## Backend Access and Migrations

SQLAlchemy 2 models live under `backend/app/models/`; database URL loading and lazy engine/session access are separated under `backend/app/config.py` and `backend/app/db/`. The API reads `DATABASE_URL` from the process environment or local `backend/.env`. Copy `backend/.env.example` to `backend/.env` and provide a real URL privately; `.env` is ignored by Git. The URL shape is:

```text
postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE
```

Alembic keeps schema changes in `backend/alembic/versions/`. From `backend/`, apply the current migration with:

```powershell
python -m alembic upgrade head
```

The initial migration enables the pgvector extension and creates `documents`, `document_chunks`, and their constraints and indexes. The extension is left installed if this migration is downgraded because it may be shared with other database objects. FastAPI startup does not create tables or run migrations.

Verify the configured database after migration with:

```powershell
python scripts/verify_database.py
```

The script checks the connection, extension, tables, vector dimension, and Alembic version table without writing business data. `GET /health/db` is a lighter connection-only check; it returns a generic failure if the database is not configured or reachable. The database URL and raw driver errors are not returned by the API.

## Document Upload Boundary

The backend accepts validated PDFs and stores them locally for development. It records the storage key, byte count, digest, and sanitized source filename in the existing document JSONB metadata. Upload status transitions through `processing` and `completed`, or to `failed` for size/storage failures. The upload workflow uses the existing schema; no database migration was needed. Local filesystem storage is not shared across production instances.

## Later RAG Work

PostgreSQL stores source records and future chunks; pgvector can later support similarity operations on actual embeddings. The embedding column remains nullable until a later phase implements and verifies embedding generation. PDF parsing, chunking, embeddings, vector indexes, keyword/hybrid retrieval, and reranking are not implemented yet.