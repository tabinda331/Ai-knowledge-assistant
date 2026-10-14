# AI Knowledge Assistant

Ask questions across PDF knowledge bases and get evidence-grounded answers with source citations.

[![CI](https://github.com/tabinda331/Ai-knowledge-assistant/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tabinda331/Ai-knowledge-assistant/actions/workflows/ci.yml)

![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4-06B6D4?logo=tailwindcss&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?logo=postgresql&logoColor=white)

[Repository](https://github.com/tabinda331/Ai-knowledge-assistant) · [Quick start](#quick-start)

## Overview

Searching long manuals, policies, and support documents by hand is slow. It is also hard to trust an answer when you cannot trace it back to evidence.

The AI Knowledge Assistant turns text-based PDFs into searchable, page-aware chunks. It combines semantic and keyword retrieval, then sends the selected passages to a configurable chat model to produce a grounded response with source references.

**NovaCart is the fictional sample business used in the demo.** Its support examples illustrate how document-grounded answers should work; they are not real-world business policies.

## Demo highlights

The fictional NovaCart knowledge base demonstrates:

| Example question | Verified demo outcome |
| --- | --- |
| What is the NovaBook Pro 14's standard warranty? | Two years, with a citation to the supporting passage. |
| How long does standard shipping take? | 3–5 business days, with a citation to the supporting passage. |
| Does NovaCart offer gaming chairs? | The assistant does not confirm an unlisted product and reports that the available documents do not provide enough evidence. |

Answers include source markers. The interface shows the source number, page reference, and chunk index so a reviewer can trace an answer to retrieved content. These examples describe only the fictional demo corpus.

## Features

- **PDF knowledge-base management:** Upload PDFs up to 25 MiB, extract text, create page-aware chunks (1,000 characters with 150 characters of overlap), generate embeddings, view processing and embedding readiness, reprocess documents, and delete document records and their stored files.
- **Dense semantic search:** Generate normalized 1,024-dimensional vectors with Sentence Transformers and search them using PostgreSQL with pgvector.
- **Keyword and hybrid retrieval:** Use BM25 keyword search alongside dense search. Hybrid mode combines rankings with Reciprocal Rank Fusion (RRF, constant 60).
- **Optional reranking:** Use a cross-encoder to reorder retrieved candidates before answer generation.
- **Grounded answers and citations:** Build answer context from retrieved passages and return structured source references. When retrieval finds no passages, the API returns a fixed insufficient-evidence response without calling the LLM. When passages exist, the model is instructed to answer from that evidence and cite its source markers.
- **OpenAI-compatible chat provider:** Configure the API key, model, and optional base URL through backend environment settings.
- **Conversation memory:** Persist user and assistant turns in PostgreSQL. Recent history is bounded by `CONVERSATION_HISTORY_MESSAGES`; earlier assistant responses are context, not evidence.
- **Search scopes:** Search one selected document or all eligible documents. Global retrieval includes completed documents whose chunks are all embedded. Document-specific BM25 search can also search processed text before embeddings are generated; dense search requires vectors.
- **Local PDF storage:** Keep uploaded PDF files on the backend filesystem while PostgreSQL stores document metadata, chunks, vectors, and conversations.

PDF processing uses text extraction; scanned PDFs that require OCR are not supported.

## Architecture

~~~mermaid
flowchart LR
    User[User] --> UI[Next.js UI]
    UI -->|Upload or ask| API[FastAPI]

    subgraph Ingestion
        API -->|Store original PDF| Files[(Local PDF storage)]
        API -->|Process request| Extract[PDF text extraction and chunking]
        Extract -->|Text and page metadata| DB[(PostgreSQL and pgvector)]
        Extract --> Embed[Sentence Transformers embeddings]
        Embed -->|Normalized 1024-dimensional vectors| DB
    end

    subgraph Answering
        API --> Dense[Dense retrieval]
        DB -->|Vector similarity| Dense
        API --> BM25[BM25 keyword retrieval]
        DB -->|Chunk text| BM25
        Dense --> RRF[Reciprocal Rank Fusion]
        BM25 --> RRF
        RRF --> Rerank[Optional cross-encoder reranking]
        Rerank --> Context[Grounded context with source markers]
        Context --> LLM[OpenAI-compatible chat model]
        LLM --> Answer[Answer with citations]
        Answer --> UI
        API <-->|Conversation history| DB
    end
~~~

## Technology stack

| Area | Technology |
| --- | --- |
| Web interface | Next.js 16, React 19, TypeScript 5, Tailwind CSS 4 |
| API | Python, FastAPI, SQLAlchemy, Alembic, psycopg |
| Database | PostgreSQL 17, pgvector |
| Embeddings | Sentence Transformers with `BAAI/bge-m3` (1,024 dimensions) |
| Reranking | Sentence Transformers CrossEncoder with `BAAI/bge-reranker-v2-m3` |
| Keyword retrieval | `rank-bm25` |
| Answer generation | OpenAI Python SDK with an OpenAI-compatible chat-completions provider |

The embedding and reranker models load locally in the backend and may download model files on first use. The chat model is configured separately through the LLM provider settings.

## Screenshots

### Warranty answer

The NovaBook Pro 14 warranty response with citations and source cards.

![Warranty answer with citations and source cards](docs/screenshots/question.png)

### Knowledge Base

NovaCart is Ready, with all 10 of 10 chunks embedded.

![Knowledge Base showing NovaCart Ready and 10 of 10 embedded chunks](docs/screenshots/knowledge-base.png)

### AI Assistant welcome screen

The NovaCart assistant welcome screen with suggested questions.

![AI Assistant welcome screen](docs/screenshots/welcome.png)

### Unsupported-product handling

The assistant responds to a question about gaming chairs using available evidence.

![Unsupported-product question about gaming chairs](docs/screenshots/unsupported-product.png)

## Quick start

### Prerequisites

- Git
- Python 3.10 or newer
- Node.js 20.9 or newer and npm
- Docker Desktop for the local PostgreSQL option

Clone the repository and enter its root directory:

~~~powershell
git clone https://github.com/tabinda331/Ai-knowledge-assistant.git
Set-Location Ai-knowledge-assistant
~~~

### 1. Start local PostgreSQL with pgvector

The local setup below binds PostgreSQL to loopback on port `55432`. If a `rag-test-postgres` container already exists and is stopped, start it with `docker start rag-test-postgres`. If it is already running, skip this step.

To create a new local container:

~~~powershell
docker run --name rag-test-postgres --env POSTGRES_DB=ragtest --env POSTGRES_USER=ragtest --env POSTGRES_PASSWORD=REPLACE_WITH_A_LOCAL_TEST_PASSWORD --publish 127.0.0.1:55432:5432 --detach pgvector/pgvector:pg17
~~~

Replace the password placeholder with a password for this local container. Do not put a real database password in the repository or a frontend environment variable.

### 2. Install backend dependencies and configure the local database

From the repository root:

~~~powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements-dev.txt
~~~

If you need a backend env file for optional local settings, copy the example only when no `backend/.env` already exists:

~~~powershell
if (-not (Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }
~~~

Do not overwrite an existing `backend/.env`. Set a process-scoped database URL in the PowerShell window used for migrations and the backend. The process environment takes precedence over the value loaded from `backend/.env`, so this selects the local container without changing an existing env file:

~~~powershell
$env:DATABASE_URL = 'postgresql+psycopg://ragtest:REPLACE_WITH_A_LOCAL_TEST_PASSWORD@127.0.0.1:55432/ragtest'
Set-Location backend
python -m alembic upgrade head
python -m alembic current
~~~

Use the local container password in the URL and URL-encode reserved characters. The environment variable remains set in that PowerShell window and its child processes. Migrations are explicit; starting FastAPI does not create or migrate tables. The initial migration needs permission to create the pgvector extension.

### 3. Start the backend

Continue in the same PowerShell window, with the local `DATABASE_URL` still set and the current directory `backend`:

~~~powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
~~~

Check the API at `http://127.0.0.1:8000/health`. `http://127.0.0.1:8000/health/db` checks connectivity to the configured database.

The backend reads these settings from its process environment or `backend/.env`:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection URL. Set it to the local URL before migrations and local API use. |
| `DOCUMENT_STORAGE_DIR` | Local PDF storage root; defaults to `backend/storage/`. |
| `EMBEDDING_MODEL` | Sentence Transformers model; defaults to `BAAI/bge-m3`. |
| `RERANKER_MODEL` | Cross-encoder model; defaults to `BAAI/bge-reranker-v2-m3`. |
| `LLM_API_KEY` | Provider key; required for `/ask` when retrieved evidence needs generation. |
| `LLM_MODEL` | Chat model; defaults to `gpt-4o-mini`. |
| `LLM_BASE_URL` | Optional OpenAI-compatible API base URL. |
| `CONVERSATION_HISTORY_MESSAGES` | Recent messages loaded for a turn; defaults to 10 and is clamped from 0 to 100. |

`HOST` and `PORT` in `backend/.env.example` are example launch settings; the application does not read them automatically. The Uvicorn command above sets the bind address and port.

### 4. Start the frontend

Open a second PowerShell window at the repository root:

~~~powershell
Set-Location frontend
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev
~~~

The frontend uses these public settings in `frontend/.env.local`:

| Variable | Purpose |
| --- | --- |
| `NEXT_PUBLIC_API_BASE_URL` | FastAPI base URL; normally `http://127.0.0.1:8000`. |
| `NEXT_PUBLIC_NOVACART_DOCUMENT_ID` | Optional ID of a fully embedded document for the NovaCart-specific chat scope. The All documents scope does not require it. |

Only put public values in `NEXT_PUBLIC_*` variables. Never put `DATABASE_URL`, `LLM_API_KEY`, or other backend credentials in frontend configuration. Open `http://localhost:3000`, choose a PDF in **Knowledge Base**, upload it, then process and embed it. Use **All documents** to search the ready documents in a fresh local database, or configure the NovaCart document ID for a document-specific scope.

The main API routes are `GET/POST /documents`, `GET/DELETE /documents/{document_id}`, `POST /documents/{document_id}/process`, `POST /documents/{document_id}/embed`, `POST /search`, and `POST /ask`.

## Verification

Run backend tests from `backend/`:

~~~powershell
Set-Location backend
python -m pytest tests -q
~~~

The backend suite uses fakes and in-memory SQLite; it does not provide automated PostgreSQL integration coverage or require a live database. The current suite has **91 passing tests**.

Run frontend checks from `frontend/`:

~~~powershell
Set-Location frontend
npm run lint
npx tsc --noEmit
npm run build
~~~

The repository has no separate TypeScript npm script; `npx tsc --noEmit` performs the type check. The first production build may need network access to retrieve the Geist fonts used through `next/font/google`.

`python scripts/verify_database.py`, run from `backend/`, checks the configured database against a seeded corpus document (136 chunks by default). It is not a generic check for a newly migrated empty database. For a fresh local database, use `python -m alembic current` and the `/health/db` endpoint.

## Limitations and roadmap

- **Local file storage:** PDFs are stored under `backend/storage/` by default. This storage is not shared between multiple backend instances.
- **Text PDFs only:** OCR for scanned documents and advanced layout/table extraction are not implemented.
- **External answer generation:** `/ask` needs a reachable, configured OpenAI-compatible LLM provider when retrieval returns evidence. Model downloads may also require network access on first use.
- **No production access controls or deployment setup:** Authentication, tenant isolation, and production deployment configuration are not included. Do not expose this local-first application publicly as-is.
- **Grounding is evidence-guided:** The prompt asks the LLM to rely on retrieved passages and cite them, but generated text is not a formal guarantee against unsupported claims.

Potential next steps include OCR and layout-aware extraction, authentication and tenant isolation, shared object storage, retrieval evaluation, and production deployment hardening.
