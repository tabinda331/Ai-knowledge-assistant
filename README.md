# AI Knowledge Assistant

An educational, portfolio-quality application that will grow into a document-grounded AI search assistant. The repository is organized as a small monorepo with a separate Next.js frontend and FastAPI backend.

## Planned Capabilities

PDF text extraction, chunking, local embedding generation, dense vector retrieval, and BM25 hybrid retrieval are implemented. Reranking, source citations, LLM chat, conversation memory, agents and tool/API integrations, evaluation, authentication, and production deployment remain future capabilities. Phase 4 did not implement OCR or retrieval.

## Technology Stack

- Frontend: Next.js, React, TypeScript, App Router, Tailwind CSS, and Vercel AI SDK (planned for a later UI phase)
- Backend: Python, FastAPI, Uvicorn, SQLAlchemy, and Alembic
- Database: PostgreSQL with pgvector; Supabase PostgreSQL is the planned hosted option
- Planned deployment: Vercel for the frontend and Render for the backend

## Repository Structure

```text
.
├── backend/       FastAPI application, SQLAlchemy models, and Alembic migrations
├── database/      Database architecture notes
├── docs/          Architecture and project documentation
├── evaluation/    Reserved for future evaluation work
├── frontend/      Next.js application
├── .gitignore
├── docker-compose.yml
└── README.md
```

## Local Setup

Prerequisites: Node.js 20 or newer, npm, and Python 3.10 or newer. Git is used for version control. Docker is not required.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. To produce a production build, run `npm run build` from `frontend/`.

### Backend

From the repository root, create and activate a virtual environment, then install the backend dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
```

Copy `backend/.env.example` to `backend/.env` and set `DATABASE_URL` to a PostgreSQL connection URL. Do not commit `backend/.env`. Example shape (replace every placeholder locally):

```text
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE
```

The API and Alembic load this file for local development; an environment variable supplied by the process takes precedence. Supabase PostgreSQL can use standard PostgreSQL connectivity, including its direct or pooler host and credentials. No Supabase SDK or account integration is required.

Start the API:

```powershell
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The same Uvicorn command works from macOS/Linux after activating the environment with `source .venv/bin/activate` from the repository root.

### Health Check

With the backend running, visit `http://127.0.0.1:8000/health` or run:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected JSON response:

```json
{"status":"ok"}
```

The frontend and backend run as separate local services. They do not make requests to one another in Phase 1. `frontend/.env.example` records the planned local API base URL for a later integration; it is not read by the current starter page.

### Database Migrations and Verification

Run commands from `backend/` with the virtual environment active and `DATABASE_URL` configured:

```powershell
python -m alembic upgrade head
python scripts/verify_database.py
```

Alembic records the applied revision in `alembic_version`. Migrations are explicit commands; FastAPI startup does not create tables or apply migrations. The verifier checks connectivity, the pgvector extension, both application tables, the `vector(1024)` column, and the migration version table. The database health endpoint checks connectivity only:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/db
```

It returns `{"status":"ok","database":"connected"}` on success and a generic HTTP 503 if no URL is configured or the database cannot be reached. It never returns the URL or the underlying database exception.

## Phase 2: Database Foundation

PostgreSQL stores the application’s structured records and provides reliable constraints, transactions, and JSONB metadata. pgvector adds a PostgreSQL vector type so embeddings can later be stored and compared alongside their source data. PostgreSQL alone does not turn text into semantic representations; future embedding generation will be needed to represent meaning as vectors, and a later retrieval phase will use those vectors to find semantically related chunks.

The `documents` table represents imported knowledge sources and their lifecycle metadata. `document_chunks` represents ordered text passages belonging to a document. Its `embedding` field is nullable `vector(1024)`, matching the planned BAAI/bge-m3 representation. It is nullable because no embeddings are generated in this phase. The unique `(document_id, chunk_index)` constraint creates a B-tree index whose leading key supports document lookups, so no duplicate single-column index is added. There is no vector similarity index yet: index choice and parameters should be based on real vectors and retrieval requirements, not an empty table.

SQLAlchemy models describe the tables to Python code; Alembic tracks schema changes as ordered, reviewable migrations. The initial migration enables pgvector and creates the tables, constraints, and document lookup index. Applying migrations is an explicit operational step rather than an application startup side effect. The backend alone reads `DATABASE_URL` and talks to PostgreSQL; the browser must call backend APIs and must never receive database credentials or connect directly.

Supabase is the planned host because it provides managed PostgreSQL, reducing the operational work of provisioning and maintaining a database while keeping PostgreSQL and pgvector available. The schema remains portable PostgreSQL and uses no Supabase-specific SDK or API; its connection is configured through the same `DATABASE_URL` variable.

## Phase 3: Document Upload

The backend accepts PDF uploads at `POST /documents` as multipart form data with a required `file` field and optional `title`. It checks the filename extension, declared content type, PDF signature, and a 25 MiB size limit. The local storage service writes atomically below `backend/storage/documents/` using generated UUID paths; it never uses the client filename as a filesystem path. The sanitized original filename, storage key, byte count, and SHA-256 digest are recorded in document metadata.

`GET /documents` lists documents with `limit` and `offset` pagination. `GET /documents/{document_id}` returns one document’s metadata and status. Uploads move through `processing` to `completed`; size or storage failures after record creation are marked `failed`. Local file storage is for development and is not shared across multiple production instances. Set `DOCUMENT_STORAGE_DIR` to choose another local root; the default `backend/storage/` is ignored by Git.

PDF parsing and chunking are implemented in Phase 4; embeddings and dense retrieval were added in later phases.

## Phase 4: PDF Parsing and Chunking

Call `POST /documents/{document_id}/process` after uploading a PDF. The backend reads the stored file, extracts text per page with `pypdf`, normalizes repeated whitespace while retaining paragraph breaks, then writes deterministic chunks to the existing `document_chunks` table. Chunk metadata records the source page range, and chunk indices start at zero for each document. Reprocessing replaces the document's existing chunks rather than adding duplicates.

The initial chunk size is 1,000 characters with 150 characters of overlap. Processing returns page, character, and chunk counts. PDFs with no usable text, unreadable PDFs, missing files, and missing documents return controlled errors; processing failures update the document to `failed` with a safe description.

This implementation handles text-based PDFs only. OCR, PDF layout analysis, and retrieval are not part of Phase 4.

## Phase 5: Embeddings

Call `POST /documents/{document_id}/embed` after processing a document. The backend loads the configured local Sentence Transformers model (default `BAAI/bge-m3`), embeds pending chunks in batches, and stores normalized 1,024-dimensional vectors in the existing nullable `document_chunks.embedding` column. The model is cached per FastAPI process; the first embedding request may download model files. Set `EMBEDDING_MODEL` in `backend/.env` to select another compatible 1,024-dimensional model. No vector search or retrieval is implemented in this phase.

The operation is repeatable: chunks with an existing embedding are skipped, and a document with no processed chunks returns HTTP 409. Run `python scripts/verify_database.py` from `backend/` to check the configured database and the known `AGENTIC RAG.pdf` document (136 chunks), including populated vector dimensions and unique chunk indexes. The verifier reports counts and dimensions only, not vector values.

## Phase 6–7: Dense and Hybrid Retrieval

Use `POST /search` with `query`, optional `top_k` (1–20, default 5), optional `document_id`, and `retrieval_mode` (`dense`, `bm25`, or `hybrid`). The default remains `dense` for backward compatibility. Every result includes chunk content and its stored page metadata. Dense mode reports cosine similarity; BM25 mode reports a lexical BM25 score and rank; hybrid mode reports dense/BM25 ranks, available component scores, and a combined score. Optional `rerank: true` reranks retrieved candidates and returns the final `top_k`; `candidate_k` (default 20, maximum 50) controls candidate generation and must be at least `top_k`.

BM25 tokenization is deterministic: text is case-folded and split into Unicode alphanumeric tokens, so punctuation and hyphens separate terms (`BGE-M3` becomes `bge`, `m3`). There is no stemming or stop-word removal. The BM25 index is cached per global or per-document scope for 60 seconds (up to 32 scopes), and successful document reprocessing invalidates it immediately. This small in-process cache fits the current 136-chunk development corpus; multiple production workers would each hold a cache, so a larger deployment should move lexical indexing to shared PostgreSQL-supported BM25 or another measured shared search service.

The live PostgreSQL 17/Supabase instance provides built-in `to_tsvector`/`ts_rank`, but `pg_search` is neither installed nor listed among available extensions. Built-in ranking is not BM25, so the current implementation uses the lightweight `rank-bm25` Python package without schema changes or a new service. Hybrid mode fuses the dense and BM25 rankings with Reciprocal Rank Fusion (RRF), using `1 / (60 + rank)` from each available list. RRF combines rank positions instead of adding incomparable cosine and BM25 raw scores. A result found by only one method remains in the fused list with just that method's rank and contribution.

## Phase 8: Cross-Encoder Reranking

Set `rerank` to `true` on `POST /search` to rerank the selected retrieval mode's candidates with `BAAI/bge-reranker-v2-m3` (configurable through `RERANKER_MODEL`). Hybrid mode is the recommended path: dense and BM25 candidate lists are fused first, then the cross-encoder scores `(query, chunk content)` pairs and returns the highest-scoring `top_k`. The score is exposed separately as `rerank_score`; original component scores, ranks, content, and metadata remain attached. Reranker scores are used for ordering only and are not added to dense, BM25, or RRF scores.

The model loads lazily, is cached per backend process, runs on CPU, and scores pairs in batches of 16. Reranking is slower than candidate retrieval, so keep `candidate_k` bounded (1–50, default 20) and set it no lower than `top_k`. The first request can take longer while model files download; subsequent requests reuse the loaded model. This stage improves the ordering signal but does not itself establish retrieval accuracy; that requires a judged evaluation set.

## Phase 9: Grounded Answers

`POST /search` remains the retrieval inspection endpoint. `POST /ask` runs the same retrieval pipeline and defaults to hybrid retrieval with reranking enabled (`top_k=5`, `candidate_k=20`). It builds source-labelled context from the final results, asks an OpenAI-compatible chat-completions API for an answer grounded only in that context, and returns the answer with structured sources made from trusted chunk IDs, document IDs, and page metadata. The prompt instructs the model to identify insufficient evidence and cite supplied source markers; this guides the model but is not a guarantee against unsupported output.

Configure `LLM_API_KEY`, optional `LLM_MODEL` (default `gpt-4o-mini`), and optional `LLM_BASE_URL` in `backend/.env`. The official OpenAI SDK uses the standard OpenAI API when no base URL is supplied and supports compatible providers when one is configured. The key is required only when `/ask` has retrieved evidence and needs generation. Retrieval with no results returns a fixed ungrounded response and does not call the model. Provider configuration/API failures return safe HTTP errors; requests have a 45-second timeout.

Tests mock the provider and require no API key or paid request. For example, run `python -m pytest tests -q` from `backend/`. Reranking and generation add latency after retrieval; CPU reranking is limited by `candidate_k`, while the LLM request is separately timeout-bounded.

## Environment Variables

`frontend/.env.example` and `backend/.env.example` contain non-secret local defaults/placeholders only. Copy values into local environment files only when needed; real `.env` files are ignored by Git. The frontend example is reserved for a future API integration. The backend loads `backend/.env` for `DATABASE_URL`, model settings, and optional LLM configuration; no credentials should be committed.

## Phase Status

**Phase 1 — Foundation: complete.** The monorepo, minimal Next.js page, and FastAPI `GET /health` endpoint are implemented and verified.

**Phase 2 — Database foundation: complete and live-verified.** PostgreSQL connectivity, pgvector, both tables, nullable `vector(1024)`, and the Alembic revision have been verified.

**Phase 3 — Document upload: complete.** PDF validation, local development storage, metadata/status persistence, listing/detail endpoints, and isolated API tests are in place.

**Phase 4 — PDF parsing and chunking: implemented and tested.** Text extraction, normalization, deterministic page-aware chunking, processing status flow, chunk persistence, and repeat-safe processing are in place. OCR and PDF layout analysis are not implemented.

**Phase 5 — Embeddings: complete and live-verified.** The existing 136 chunks for `AGENTIC RAG.pdf` have BAAI/bge-m3 vectors stored in PostgreSQL; database verification confirms every vector is 1,024-dimensional.

**Phase 6 — Dense vector retrieval: complete and live-verified.** `POST /search` embeds the query with the existing BGE-M3 service, orders matching rows by pgvector cosine distance, and returns bounded top-K evidence with chunk metadata and cosine similarity.

**Phase 7 — BM25 and hybrid retrieval: complete and live-verified.** `POST /search` supports dense, BM25, and RRF hybrid modes with optional document filtering. Results are evidence only.

**Phase 8 — Cross-encoder reranking: implemented.** Search can rerank a bounded candidate set after retrieval and preserve original evidence metadata and scores. No answer generation is implemented.

**Phase 9 — Grounded answer generation: implemented; live provider verification depends on local LLM configuration.** `/ask` reuses retrieval/reranking, builds context from final chunks, calls an OpenAI-compatible provider, and constructs structured sources from retrieved metadata. Tests use a mocked provider.

## Future Development Phases

1. Add conversation workflows.
2. Add tool integrations, evaluation, observability, and deployment.

These are roadmap items, not current features. Later phases will begin only after approval.