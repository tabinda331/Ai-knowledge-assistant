"use client";

import { useEffect, useState, type FormEvent } from "react";

import AppHeader from "@/app/components/app-header";
import {
  deleteDocument,
  embedDocument,
  listDocuments,
  processDocument,
  uploadDocument,
  type DocumentResponse,
} from "@/lib/api";

const PAGE_SIZE = 50;

const lifecycleLabels = {
  uploaded: "Uploaded",
  processed: "Processed",
  ready: "Ready",
  failed: "Failed",
} as const;

type DocumentAction = "process" | "embed" | "delete";
type DocumentFeedback = { kind: "success" | "error"; message: string };

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Date unavailable"
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(date);
}

export default function KnowledgeBaseWorkspace() {
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [activeActions, setActiveActions] = useState<Record<string, DocumentAction>>({});
  const [documentFeedback, setDocumentFeedback] = useState<Record<string, DocumentFeedback>>({});
  const [pageNotice, setPageNotice] = useState<DocumentFeedback | null>(null);

  async function loadFirstPage() {
    setLoading(true);
    setListError(null);
    try {
      const result = await listDocuments({ limit: PAGE_SIZE, offset: 0 });
      setDocuments(result);
      setOffset(result.length);
      setHasMore(result.length === PAGE_SIZE);
    } catch (error) {
      setListError(error instanceof Error ? error.message : "Could not load documents.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let active = true;
    listDocuments({ limit: PAGE_SIZE, offset: 0 })
      .then((result) => {
        if (!active) return;
        setDocuments(result);
        setOffset(result.length);
        setHasMore(result.length === PAGE_SIZE);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setListError(error instanceof Error ? error.message : "Could not load documents.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  async function loadMore() {
    setLoadingMore(true);
    setListError(null);
    try {
      const result = await listDocuments({ limit: PAGE_SIZE, offset });
      setDocuments((current) => [...current, ...result]);
      setOffset((current) => current + result.length);
      setHasMore(result.length === PAGE_SIZE);
    } catch (error) {
      setListError(error instanceof Error ? error.message : "Could not load more documents.");
    } finally {
      setLoadingMore(false);
    }
  }

  async function refreshLoadedDocuments(targetCount: number): Promise<boolean> {
    const refreshed: DocumentResponse[] = [];
    let nextOffset = 0;
    let hasNextPage = true;
    try {
      while (hasNextPage && (refreshed.length < targetCount || refreshed.length === 0)) {
        const page = await listDocuments({ limit: PAGE_SIZE, offset: nextOffset });
        refreshed.push(...page);
        nextOffset += page.length;
        hasNextPage = page.length === PAGE_SIZE;
      }
      setDocuments(refreshed);
      setOffset(refreshed.length);
      setHasMore(hasNextPage);
      setListError(null);
      return true;
    } catch (error) {
      setListError(error instanceof Error ? error.message : "Could not refresh documents.");
      return false;
    }
  }

  function setActionFeedback(documentId: string, feedback: DocumentFeedback) {
    setDocumentFeedback((current) => ({ ...current, [documentId]: feedback }));
  }

  function startDocumentAction(documentId: string, action: DocumentAction) {
    setActiveActions((current) => ({ ...current, [documentId]: action }));
    setDocumentFeedback((current) => {
      const next = { ...current };
      delete next[documentId];
      return next;
    });
  }

  function finishDocumentAction(documentId: string) {
    setActiveActions((current) => {
      const next = { ...current };
      delete next[documentId];
      return next;
    });
  }

  async function handleProcess(item: DocumentResponse) {
    startDocumentAction(item.id, "process");
    try {
      const result = await processDocument(item.id);
      setDocuments((current) => current.map((document) => (
        document.id === item.id
          ? {
              ...document,
              chunk_count: result.chunks_created,
              embedded_chunk_count: 0,
              lifecycle_state: result.chunks_created > 0 ? "processed" : "uploaded",
              metadata: {
                ...document.metadata,
                processing_stats: {
                  pages_processed: result.pages_processed,
                  characters_extracted: result.characters_extracted,
                  chunks_created: result.chunks_created,
                  chunk_size: result.chunk_size,
                  chunk_overlap: result.chunk_overlap,
                },
                processing_error: undefined,
              },
            }
          : document
      )));
      setActionFeedback(item.id, {
        kind: "success",
        message: `Processing completed: ${result.pages_processed} pages and ${result.chunks_created} chunks.`,
      });
      const refreshed = await refreshLoadedDocuments(documents.length);
      if (!refreshed) {
        setActionFeedback(item.id, {
          kind: "success",
          message: "Processing completed. The refreshed counts are shown, but the document list could not be reloaded.",
        });
      }
    } catch (error) {
      setActionFeedback(item.id, {
        kind: "error",
        message: error instanceof Error ? error.message : "Document processing failed.",
      });
      await refreshLoadedDocuments(documents.length);
    } finally {
      finishDocumentAction(item.id);
    }
  }

  async function handleEmbedding(item: DocumentResponse) {
    startDocumentAction(item.id, "embed");
    try {
      const result = await embedDocument(item.id);
      setDocuments((current) => current.map((document) => (
        document.id === item.id
          ? { ...document, embedded_chunk_count: result.chunks_embedded }
          : document
      )));
      setActionFeedback(item.id, {
        kind: "success",
        message: `Embedding completed for ${result.chunks_embedded} chunks.`,
      });
      const refreshed = await refreshLoadedDocuments(documents.length);
      if (!refreshed) {
        setActionFeedback(item.id, {
          kind: "success",
          message: "Embedding completed. Counts from the response are shown, but the document list could not be reloaded.",
        });
      }
    } catch (error) {
      setActionFeedback(item.id, {
        kind: "error",
        message: error instanceof Error ? error.message : "Embedding generation failed.",
      });
      await refreshLoadedDocuments(documents.length);
    } finally {
      finishDocumentAction(item.id);
    }
  }

  async function handleDelete(item: DocumentResponse) {
    const confirmed = window.confirm(
      `Delete “${item.title}”? The document, its chunks, embeddings, and stored PDF will be removed.`,
    );
    if (!confirmed) return;

    setPageNotice(null);
    startDocumentAction(item.id, "delete");
    try {
      await deleteDocument(item.id);
      const remainingCount = Math.max(0, documents.length - 1);
      setDocuments((current) => current.filter((document) => document.id !== item.id));
      setDocumentFeedback((current) => {
        const next = { ...current };
        delete next[item.id];
        return next;
      });
      const refreshed = await refreshLoadedDocuments(remainingCount);
      setPageNotice({
        kind: refreshed ? "success" : "error",
        message: refreshed
          ? `${item.title} was deleted.`
          : `${item.title} was deleted, but the document list could not be refreshed.`,
      });
    } catch (error) {
      setActionFeedback(item.id, {
        kind: "error",
        message: error instanceof Error ? error.message : "Document deletion failed.",
      });
      const refreshed = await refreshLoadedDocuments(documents.length);
      if (refreshed) {
        setPageNotice({
          kind: "error",
          message: "The delete request reported an error. The document list was refreshed to show its current state.",
        });
      }
    } finally {
      finishDocumentAction(item.id);
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setUploadError(null);
    setUploadSuccess(null);
    if (!file) {
      setUploadError("Choose a PDF file to upload.");
      return;
    }

    setUploading(true);
    try {
      const created = await uploadDocument(file, title);
      setDocuments((current) => [created, ...current.filter((item) => item.id !== created.id)]);
      setFile(null);
      setTitle("");
      setUploadSuccess(`${created.filename ?? created.title} was uploaded.`);
      const input = document.getElementById("knowledge-base-file") as HTMLInputElement | null;
      if (input) input.value = "";
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Could not upload this PDF.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="kb-app">
      <AppHeader active="knowledge-base" />
      <main className="kb-main">
        {pageNotice ? (
          <p className={`kb-alert ${pageNotice.kind === "error" ? "kb-alert-error" : "kb-alert-success"}`} role={pageNotice.kind === "error" ? "alert" : "status"}>
            {pageNotice.message}
          </p>
        ) : null}
        <section className="kb-page-heading">
          <div>
            <p className="kb-eyebrow">WORKSPACE</p>
            <h1>Knowledge Base</h1>
            <p className="kb-subtitle">Manage the documents that support your AI assistant.</p>
          </div>
          {!loading && !listError ? (
            <div className="kb-total" aria-live="polite">
              <span className="kb-total-value">{documents.length}{hasMore ? "+" : ""}</span>
              <span>documents loaded</span>
            </div>
          ) : null}
        </section>

        <section className="kb-upload-panel" aria-labelledby="upload-heading">
          <div className="kb-panel-heading">
            <div className="kb-upload-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14.5v3A2.5 2.5 0 0 0 7.5 20h9a2.5 2.5 0 0 0 2.5-2.5v-3" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <div>
              <h2 id="upload-heading">Add a document</h2>
              <p>Upload a PDF to add it to your knowledge base.</p>
            </div>
          </div>

          <form className="kb-upload-form" onSubmit={handleUpload}>
            <label className="kb-file-picker" htmlFor="knowledge-base-file">
              <span className="kb-file-picker-icon" aria-hidden="true">PDF</span>
              <span className="kb-file-picker-copy">
                <strong>{file ? file.name : "Choose a PDF to upload"}</strong>
                <span>{file ? formatBytes(file.size) : "PDF files up to 25 MiB"}</span>
              </span>
              <span className="kb-browse-button">Browse files</span>
              <input
                id="knowledge-base-file"
                type="file"
                accept="application/pdf,.pdf"
                onChange={(event) => {
                  setFile(event.currentTarget.files?.[0] ?? null);
                  setUploadError(null);
                  setUploadSuccess(null);
                }}
                disabled={uploading}
              />
            </label>
            <label className="kb-title-field">
              <span>Document title <span>(optional)</span></span>
              <input
                value={title}
                onChange={(event) => setTitle(event.currentTarget.value)}
                maxLength={500}
                placeholder="Use the file name"
                disabled={uploading}
              />
            </label>
            <div className="kb-upload-actions">
              <p>Documents are processed and embedded as separate steps.</p>
              <button className="kb-primary-button" type="submit" disabled={uploading || !file}>
                {uploading ? <><span className="kb-spinner" aria-hidden="true" /> Uploading…</> : "Upload PDF"}
              </button>
            </div>
          </form>
          {uploadError ? <p className="kb-alert kb-alert-error" role="alert">{uploadError}</p> : null}
          {uploadSuccess ? <p className="kb-alert kb-alert-success" role="status">{uploadSuccess}</p> : null}
        </section>

        <section className="kb-documents-section" aria-labelledby="documents-heading">
          <div className="kb-list-heading">
            <div>
              <p className="kb-eyebrow">YOUR LIBRARY</p>
              <h2 id="documents-heading">Documents</h2>
            </div>
            <button className="kb-refresh-button" type="button" onClick={() => void loadFirstPage()} disabled={loading || loadingMore}>
              <svg aria-hidden="true" viewBox="0 0 20 20" fill="none">
                <path d="M16 8a6 6 0 0 0-10.4-2L4 7.5M4 4v3.5h3.5M4 12a6 6 0 0 0 10.4 2l1.6-1.5M16 16v-3.5h-3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Refresh
            </button>
          </div>

          {loading ? (
            <div className="kb-state-card" role="status">
              <span className="kb-spinner kb-spinner-large" aria-hidden="true" />
              <strong>Loading documents</strong>
              <span>Connecting to your knowledge base…</span>
            </div>
          ) : listError && documents.length === 0 ? (
            <div className="kb-state-card kb-state-error" role="alert">
              <span className="kb-state-symbol" aria-hidden="true">!</span>
              <strong>Documents could not be loaded</strong>
              <span>{listError}</span>
              <button className="kb-secondary-button" type="button" onClick={() => void loadFirstPage()}>Try again</button>
            </div>
          ) : documents.length === 0 ? (
            <div className="kb-state-card">
              <span className="kb-empty-symbol" aria-hidden="true">
                <svg viewBox="0 0 28 28" fill="none">
                  <path d="M7 3.5h9l6 6v15H7a2 2 0 0 1-2-2v-17a2 2 0 0 1 2-2Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
                  <path d="M16 3.5v6h6M10 15h8M10 19h8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
              <strong>Your library is ready for its first document</strong>
              <span>Upload a PDF above to start building your knowledge base.</span>
            </div>
          ) : (
            <div className="kb-document-list">
              {documents.map((item) => {
                const stats = item.metadata.processing_stats;
                const displayName = item.filename ?? item.metadata.original_filename ?? item.title;
                return (
                  <article className="kb-document-card" key={item.id}>
                    <div className="kb-document-main">
                      <span className="kb-document-icon" aria-hidden="true">PDF</span>
                      <div className="kb-document-copy">
                        <h3>{item.title}</h3>
                        <p>{displayName}</p>
                      </div>
                      <span className={`kb-status kb-status-${item.lifecycle_state}`}>
                        <span aria-hidden="true" />{lifecycleLabels[item.lifecycle_state]}
                      </span>
                    </div>
                    <div className="kb-document-stats">
                      <div className="kb-stat">
                        <span>Chunks</span>
                        <strong>{item.chunk_count}</strong>
                      </div>
                      <div className="kb-stat">
                        <span>Embedded</span>
                        <strong>{item.embedded_chunk_count}<span> / {item.chunk_count}</span></strong>
                      </div>
                      {typeof stats?.pages_processed === "number" ? (
                        <div className="kb-stat">
                          <span>Pages</span>
                          <strong>{stats.pages_processed}</strong>
                        </div>
                      ) : null}
                      {typeof stats?.characters_extracted === "number" ? (
                        <div className="kb-stat">
                          <span>Characters</span>
                          <strong>{stats.characters_extracted.toLocaleString()}</strong>
                        </div>
                      ) : null}
                      {typeof stats?.chunk_size === "number" ? (
                        <div className="kb-stat">
                          <span>Chunk size</span>
                          <strong>{stats.chunk_size.toLocaleString()} chars</strong>
                        </div>
                      ) : null}
                      {typeof stats?.chunk_overlap === "number" ? (
                        <div className="kb-stat">
                          <span>Overlap</span>
                          <strong>{stats.chunk_overlap.toLocaleString()} chars</strong>
                        </div>
                      ) : null}
                      {typeof item.metadata.size_bytes === "number" ? (
                        <div className="kb-stat">
                          <span>File size</span>
                          <strong>{formatBytes(item.metadata.size_bytes)}</strong>
                        </div>
                      ) : null}
                      <div className="kb-stat kb-stat-date">
                        <span>Added</span>
                        <strong>{formatDate(item.created_at)}</strong>
                      </div>
                    </div>
                    {item.lifecycle_state === "failed" && item.metadata.processing_error ? (
                      <p className="kb-document-error">{item.metadata.processing_error}</p>
                    ) : null}
                    <div className="kb-document-actions">
                      {item.lifecycle_state === "uploaded" ||
                      item.lifecycle_state === "processed" ||
                      item.lifecycle_state === "ready" ||
                      (item.lifecycle_state === "failed" && Boolean(item.metadata.processing_error)) ? (
                        <button
                          className="kb-secondary-button"
                          type="button"
                          onClick={() => void handleProcess(item)}
                          disabled={Boolean(activeActions[item.id])}
                        >
                          {activeActions[item.id] === "process" ? (
                            <><span className="kb-spinner" aria-hidden="true" /> Processing…</>
                          ) : item.chunk_count > 0 ? "Reprocess" : "Process"}
                        </button>
                      ) : null}
                      {item.lifecycle_state === "processed" &&
                      item.chunk_count > 0 &&
                      item.embedded_chunk_count < item.chunk_count ? (
                        <button
                          className="kb-secondary-button"
                          type="button"
                          onClick={() => void handleEmbedding(item)}
                          disabled={Boolean(activeActions[item.id])}
                        >
                          {activeActions[item.id] === "embed" ? (
                            <><span className="kb-spinner" aria-hidden="true" /> Generating…</>
                          ) : "Generate embeddings"}
                        </button>
                      ) : null}
                      <button
                        className="kb-danger-button"
                        type="button"
                        onClick={() => void handleDelete(item)}
                        disabled={Boolean(activeActions[item.id])}
                      >
                        {activeActions[item.id] === "delete" ? (
                          <><span className="kb-spinner" aria-hidden="true" /> Deleting…</>
                        ) : "Delete"}
                      </button>
                    </div>
                    {documentFeedback[item.id] ? (
                      <p
                        className={`kb-action-feedback ${documentFeedback[item.id].kind === "error" ? "is-error" : "is-success"}`}
                        role={documentFeedback[item.id].kind === "error" ? "alert" : "status"}
                      >
                        {documentFeedback[item.id].message}
                      </p>
                    ) : null}
                  </article>
                );
              })}
            </div>
          )}
          {listError && !loading && documents.length > 0 ? (
            <p className="kb-alert kb-alert-error" role="alert">{listError}</p>
          ) : null}
          {hasMore && !loading && !listError ? (
            <button className="kb-load-more" type="button" onClick={() => void loadMore()} disabled={loadingMore}>
              {loadingMore ? <><span className="kb-spinner" aria-hidden="true" /> Loading…</> : "Load more documents"}
            </button>
          ) : null}
        </section>
      </main>
      <footer className="page-footer">
        <span>Manage the sources behind NovaCart’s AI assistant.</span>
        <span>KNOWLEDGE BASE</span>
      </footer>
    </div>
  );
}
