import AnswerContent, { sourceElementId } from "@/app/components/answer-content";
import type { ChatSource } from "@/lib/api";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  grounded?: boolean;
  retrieval_mode?: string;
  reranked?: boolean;
};

function pageLabel(source: ChatSource): string {
  const { page_start: start, page_end: end } = source;

  if (start == null && end == null) return "Pages unavailable";
  if (start == null) return `Page ${end}`;
  if (end == null || end === start) return `Page ${start}`;
  return `Pages ${start}–${end}`;
}

function SourceList({ messageId, sources }: { messageId: string; sources: ChatSource[] }) {
  if (sources.length === 0) return null;

  return (
    <section className="sources-section" aria-label="Answer sources">
      <h3>Sources</h3>
      <ol className="source-list">
        {sources.map((source, index) => (
          <li
            className="source-item"
            id={sourceElementId(messageId, index + 1)}
            key={`${source.chunk_id}-${index}`}
            tabIndex={-1}
          >
            <span className="source-number" aria-hidden="true">{index + 1}</span>
            <span className="source-copy">
              <strong>Source {index + 1}</strong>
              <span>{pageLabel(source)} <span aria-hidden="true">·</span> Chunk {source.chunk_index}</span>
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

export default function ChatMessageView({ message }: { message: ChatMessage }) {
  const sources = message.sources ?? [];

  return (
    <article className={`message-card message-card--${message.role}`}>
      <div className="message-author">
        {message.role === "assistant" ? (
          <span className="assistant-mark" aria-hidden="true">N</span>
        ) : null}
        <span>{message.role === "user" ? "You" : "NovaCart AI"}</span>
      </div>

      {message.role === "user" ? (
        <p className="user-message-content">{message.content}</p>
      ) : (
        <>
          <AnswerContent content={message.content} messageId={message.id} sources={sources} />

          <p className={`grounding-status ${message.grounded ? "is-grounded" : "is-limited"}`}>
            <span className="grounding-dot" aria-hidden="true" />
            {message.grounded
              ? "Answer grounded in knowledge base"
              : "Limited knowledge-base evidence"}
          </p>

          <SourceList messageId={message.id} sources={sources} />

          {message.retrieval_mode ? (
            <details className="technical-details">
              <summary>Technical details</summary>
              <p>
                {message.retrieval_mode}
                {message.reranked ? " · reranked" : ""}
              </p>
            </details>
          ) : null}
        </>
      )}
    </article>
  );
}
