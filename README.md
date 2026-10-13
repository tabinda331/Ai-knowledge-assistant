# AI Knowledge Assistant

A local-first document assistant with a FastAPI backend, PostgreSQL/pgvector storage, and a Next.js interface.

## Current capabilities

- Upload, list, inspect, process, embed, and delete PDF documents.
- Extract text and create ordered, page-aware chunks.
- Generate normalized 1,024-dimensional embeddings with a local Sentence Transformers model.
- Search with dense vector, BM25, or hybrid retrieval; hybrid search combines rankings with reciprocal rank fusion.
- Optionally rerank candidates with a cross-encoder.
- Generate evidence-grounded answers with citations through an OpenAI-compatible chat API.
- Persist conversation turns and bounded history in PostgreSQL.
- Use the Knowledge Base and chat interfaces in the frontend.

PDF files are stored locally; PostgreSQL stores document metadata, chunks, embeddings, and conversations. OCR is not implemented. Local file storage is intended for development and a single backend instance.

## Stack

- Backend: Python, FastAPI, SQLAlchemy, Alembic, psycopg
- Database: PostgreSQL 17 with pgvector
- Frontend: Next.js, React, TypeScript
- Local document models: Sentence Transformers

## Windows local setup

Prerequisites: Git, Python 3.10 or newer, Node.js 20 or newer, npm, and Docker Desktop for the local PostgreSQL option.

### 1. Start local PostgreSQL with pgvector

If the disposable rag-test-postgres container already exists and is running, skip this command. To create a new local-only container:

~~~powershell
docker run --name rag-test-postgres --env POSTGRES_DB=ragtest --env POSTGRES_USER=ragtest --env POSTGRES_PASSWORD=REPLACE_WITH_A_LOCAL_TEST_PASSWORD --publish 127.0.0.1:55432:5432 --detach pgvector/pgvector:pg17
~~~

The port is bound to loopback. Use the password configured for that container; the value above is a placeholder, not a project credential.

### 2. Install backend dependencies

Run from the repository root:

~~~powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements-dev.txt
~~~

The development requirements include the runtime packages, pytest, and HTTPX.

### 3. Point migrations and the backend at the local database

The backend loads backend/.env when configuration is imported. A process environment variable takes precedence over the file. To avoid changing an existing .env that may point elsewhere, set the URL only in a dedicated PowerShell window:

If you need a local backend env file for non-database settings, create it only when one does not already exist:

~~~powershell
if (-not (Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }
~~~

Do not overwrite an existing backend/.env. The process-scoped URL below selects the disposable database without changing it.

~~~powershell
$env:DATABASE_URL = 'postgresql+psycopg://ragtest:REPLACE_WITH_A_LOCAL_TEST_PASSWORD@127.0.0.1:55432/ragtest'
Set-Location backend
python -m alembic upgrade head
python -m alembic current
~~~

Replace the placeholder with the disposable container's local password. URL-encode reserved characters in the password. The process variable lasts in that PowerShell session and its child processes; closing the window clears it. Do not put database credentials in frontend environment files.

Migrations are explicit commands. Starting FastAPI does not create tables or run migrations.

### 4. Start the backend

In the same PowerShell window, with DATABASE_URL still set and the current directory backend:

~~~powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
~~~

Check http://127.0.0.1:8000/health. The database check at http://127.0.0.1:8000/health/db opens a connection to the configured database.

The backend also reads these optional settings from its process environment or backend/.env:

| Variable | Purpose |
| --- | --- |
| DOCUMENT_STORAGE_DIR | PDF storage root; defaults to backend/storage/. |
| EMBEDDING_MODEL | Sentence Transformers embedding model; defaults to BAAI/bge-m3. |
| RERANKER_MODEL | Cross-encoder model; defaults to BAAI/bge-reranker-v2-m3. |
| LLM_API_KEY | Provider key, required only when /ask has evidence to send for generation. |
| LLM_MODEL | Chat model; defaults to gpt-4o-mini. |
| LLM_BASE_URL | Optional OpenAI-compatible API base URL. |
| CONVERSATION_HISTORY_MESSAGES | Maximum messages loaded for a turn; defaults to 10 and is clamped from 0 to 100. |

HOST and PORT in backend/.env.example are example launch settings; the application does not read them automatically. Pass bind address and port to Uvicorn as shown.

### 5. Start the frontend

In a second PowerShell window:

~~~powershell
Set-Location frontend
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev
~~~

Set only public frontend values in frontend/.env.local:

| Variable | Purpose |
| --- | --- |
| NEXT_PUBLIC_API_BASE_URL | Backend URL, normally http://127.0.0.1:8000. |
| NEXT_PUBLIC_NOVACART_DOCUMENT_ID | Document ID used by the default NovaCart chat scope; set it to a fully embedded document. |

The chat can also use the all-documents scope. Never place DATABASE_URL, LLM_API_KEY, or other backend secrets in a NEXT_PUBLIC_* variable. Open http://localhost:3000.

### Upload and test a document

Use the Knowledge Base page to upload a PDF, process it, then embed it. The chat and /search endpoint can retrieve fully embedded documents. Global retrieval excludes documents with missing embeddings; an explicit document filter can search the embedded chunks of a partially embedded document.

The main API routes are GET/POST /documents, GET/DELETE /documents/{id}, POST /documents/{id}/process, POST /documents/{id}/embed, POST /search, and POST /ask.

## Verification commands

Backend tests use fake sessions or in-memory SQLite and do not require a live PostgreSQL database:

~~~powershell
Set-Location backend
python -m pytest tests -q
~~~

Frontend checks, from frontend/:

~~~powershell
npm run lint
npx tsc --noEmit
npm run build
~~~

There is no separate TypeScript npm script; npx tsc --noEmit performs the type check. The production build uses next/font/google and therefore needs network access to fetch Geist fonts unless they are already available to Next.js.

python scripts/verify_database.py is a corpus-specific verifier: by default it expects a known document with 136 chunks and populated embeddings. It is not a generic check for a fresh empty database. For a newly migrated database, use python -m alembic current and /health/db.

## Safety and limitations

- Keep real backend credentials in backend/.env or process environment; .env files are ignored by Git.
- Use a process-scoped local DATABASE_URL before running migrations or database-backed manual API requests. The browser must only receive the backend API URL.
- Document deletion removes its database rows and stored PDF.
- Embedding and reranking models may download on first use. /ask with evidence requires a reachable configured LLM provider.
- Local PDF storage is not shared across multiple backend instances. OCR, authentication, and production deployment are not included.
