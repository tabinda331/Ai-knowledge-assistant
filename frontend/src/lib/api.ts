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
  document_id: string;
};

export type AskResponse = {
  question: string;
  answer: string;
  sources: ChatSource[];
  grounded: boolean;
  retrieval_mode: string;
  reranked: boolean;
  conversation_id: string;
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

export async function askQuestion(
  query: string,
  conversationId?: string,
): Promise<AskResponse> {
  const payload: AskRequest = {
    query,
    retrieval_mode: "hybrid",
    top_k: 5,
    candidate_k: 20,
    rerank: true,
    document_id: getDocumentId(),
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
