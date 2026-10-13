export type ChatSource = {
  chunk_id: string;
  document_id: string;
  chunk_index: number;
  page_start: number | null;
  page_end: number | null;
};

export type AskRequest = {
  query: string;
  conversation_id?: string;
  retrieval_mode: "hybrid";
  top_k: 5;
  candidate_k: 20;
  rerank: true;
  document_id?: string;
};

export type ChatRetrievalScope = "novacart" | "all";

export type AskResponse = {
  question: string;
  answer: string;
  sources: ChatSource[];
  grounded: boolean;
  retrieval_mode: string;
  reranked: boolean;
  conversation_id: string;
};

export type DocumentLifecycle = "uploaded" | "processed" | "ready" | "failed";

export type DocumentProcessingStats = {
  pages_processed?: number;
  characters_extracted?: number;
  chunks_created?: number;
  chunk_size?: number;
  chunk_overlap?: number;
};

export type DocumentMetadata = {
  original_filename?: string;
  storage_key?: string;
  size_bytes?: number;
  sha256?: string;
  processing_stats?: DocumentProcessingStats;
  processing_error?: string;
  failure_code?: string;
  [key: string]: unknown;
};

export type DocumentResponse = {
  id: string;
  title: string;
  filename: string | null;
  source_type: string;
  source_url: string | null;
  mime_type: string | null;
  status: string;
  metadata: DocumentMetadata;
  created_at: string;
  updated_at: string;
  chunk_count: number;
  embedded_chunk_count: number;
  lifecycle_state: DocumentLifecycle;
};

export type ListDocumentsParams = {
  limit?: number;
  offset?: number;
};

export type ProcessDocumentResponse = {
  document_id: string;
  status: string;
  pages_processed: number;
  characters_extracted: number;
  chunks_created: number;
  chunk_size: number;
  chunk_overlap: number;
};

export type EmbedDocumentResponse = {
  document_id: string;
  status: string;
  chunks_embedded: number;
  embedding_dimension: number;
  model: string;
};

export type DeleteDocumentResponse = {
  id: string;
  status: "deleted";
};

function getApiBaseUrl(): string {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (!baseUrl) {
    throw new Error("Set NEXT_PUBLIC_API_BASE_URL in the frontend environment.");
  }
  return baseUrl.replace(/\/+$/, "");
}

function getDocumentId(): string {
  const documentId = process.env.NEXT_PUBLIC_NOVACART_DOCUMENT_ID?.trim();
  if (!documentId) {
    throw new Error("Set NEXT_PUBLIC_NOVACART_DOCUMENT_ID in the frontend environment.");
  }
  return documentId;
}

async function requestDocumentApi<T>(path: string, init?: RequestInit): Promise<T> {
  const apiUrl = `${getApiBaseUrl()}${path}`;
  let response: Response;
  try {
    response = await fetch(apiUrl, init);
  } catch {
    throw new Error("Could not reach the document service. Check that the backend is running.");
  }

  if (!response.ok) {
    let detail = `The document request failed (${response.status}).`;
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null && "detail" in body) {
        const value = body.detail;
        if (typeof value === "string") {
          detail = value;
        } else if (Array.isArray(value)) {
          detail = value
            .map((item) => {
              if (typeof item === "object" && item !== null && "msg" in item && typeof item.msg === "string") {
                return item.msg;
              }
              return null;
            })
            .filter((message): message is string => message !== null)
            .join(" ") || detail;
        }
      }
    } catch {
      // Retain the status-based message when the backend response is not JSON.
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export async function listDocuments(
  params: ListDocumentsParams = {},
): Promise<DocumentResponse[]> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));
  const queryString = query.toString();
  const suffix = queryString ? `?${queryString}` : "";
  return requestDocumentApi<DocumentResponse[]>(`/documents${suffix}`);
}

export async function getDocument(documentId: string): Promise<DocumentResponse> {
  return requestDocumentApi<DocumentResponse>(`/documents/${encodeURIComponent(documentId)}`);
}

export async function uploadDocument(file: File, title?: string): Promise<DocumentResponse> {
  const formData = new FormData();
  formData.set("file", file);
  if (title?.trim()) formData.set("title", title.trim());
  return requestDocumentApi<DocumentResponse>("/documents", {
    method: "POST",
    body: formData,
  });
}

export async function processDocument(documentId: string): Promise<ProcessDocumentResponse> {
  return requestDocumentApi<ProcessDocumentResponse>(
    `/documents/${encodeURIComponent(documentId)}/process`,
    { method: "POST" },
  );
}

export async function embedDocument(documentId: string): Promise<EmbedDocumentResponse> {
  return requestDocumentApi<EmbedDocumentResponse>(
    `/documents/${encodeURIComponent(documentId)}/embed`,
    { method: "POST" },
  );
}

export async function deleteDocument(documentId: string): Promise<DeleteDocumentResponse> {
  return requestDocumentApi<DeleteDocumentResponse>(
    `/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  );
}

export async function askQuestion(
  query: string,
  conversationId?: string,
  scope: ChatRetrievalScope = "novacart",
): Promise<AskResponse> {
  const payload: AskRequest = {
    query,
    retrieval_mode: "hybrid",
    top_k: 5,
    candidate_k: 20,
    rerank: true,
    ...(scope === "novacart" ? { document_id: getDocumentId() } : {}),
    ...(conversationId ? { conversation_id: conversationId } : {}),
  };
  const apiUrl = `${getApiBaseUrl()}/ask`;

  let response: Response;
  try {
    response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error("Could not reach the knowledge assistant. Check that the backend is running.");
  }

  if (!response.ok) {
    let detail = `The request failed (${response.status}).`;
    try {
      const body: unknown = await response.json();
      if (
        typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof body.detail === "string"
      ) {
        detail = body.detail;
      }
    } catch {
      // Keep the status-based message when the backend response is not JSON.
    }
    throw new Error(detail);
  }

  return (await response.json()) as AskResponse;
}
