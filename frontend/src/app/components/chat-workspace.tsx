"use client";

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import ChatMessageView, { type ChatMessage } from "@/app/components/chat-message";
import { askQuestion } from "@/lib/api";

const suggestedQuestions = [
  "What is the warranty for the NovaBook Pro 14?",
  "How long does standard shipping take?",
  "What is the return policy?",
  "Does NovaCart offer express shipping?",
];

type LoadingStage = "searching" | "preparing";

export default function ChatWorkspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [pendingTurn, setPendingTurn] = useState<ChatMessage | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState<LoadingStage>("searching");
  const [error, setError] = useState(false);
  const endOfMessages = useRef<HTMLDivElement>(null);
  const hasMessages = messages.length > 0 || pendingTurn !== null;

  useEffect(() => {
    endOfMessages.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pendingTurn, loading, error]);

  useEffect(() => {
    if (!loading) return;

    const timer = window.setTimeout(() => setLoadingStage("preparing"), 1400);
    return () => window.clearTimeout(timer);
  }, [loading]);

  async function sendTurn(turn: ChatMessage) {
    if (loading) return;
    setPendingTurn(turn);
    setError(false);
    setLoadingStage("searching");
    setLoading(true);

    try {
      const response = await askQuestion(turn.content, conversationId ?? undefined);
      setConversationId(response.conversation_id);
      setMessages((current) => [
        ...current,
        turn,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          grounded: response.grounded,
          retrieval_mode: response.retrieval_mode,
          reranked: response.reranked,
        },
      ]);
      setPendingTurn(null);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }

  function startTurn(question: string) {
    if (loading || (error && pendingTurn)) return;
    setInput("");
    void sendTurn({ id: crypto.randomUUID(), role: "user", content: question });
  }

  function submitMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = input.trim();
    if (!question || loading) return;
    startTurn(question);
  }

  function handleInputKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  function startNewChat() {
    if (loading) return;
    setMessages([]);
    setInput("");
    setConversationId(null);
    setPendingTurn(null);
    setError(false);
  }

  const visibleMessages = pendingTurn ? [...messages, pendingTurn] : messages;
  const inputDisabled = loading || Boolean(error && pendingTurn);

  return (
    <div className="chat-app">
      <header className="topbar">
        <div className="brand" aria-label="NovaCart AI Customer Assistant">
          <span className="brand-mark" aria-hidden="true">N</span>
          <span className="brand-copy">
            <strong>NovaCart</strong>
            <span>AI Customer Assistant</span>
          </span>
        </div>

        <div className="header-actions">
          <span className="knowledge-badge">
            <span className="knowledge-dot" aria-hidden="true" />
            NovaCart knowledge base
          </span>
          <button
            className="new-chat-button"
            type="button"
            onClick={startNewChat}
            disabled={loading}
          >
            <svg aria-hidden="true" viewBox="0 0 20 20" fill="none">
              <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
            </svg>
            New Chat
          </button>
        </div>
      </header>

      <main className="chat-main">
        <section className="conversation-panel" aria-label="NovaCart conversation">
          {!hasMessages ? (
            <div className="welcome-state">
              <span className="welcome-mark" aria-hidden="true">N</span>
              <p className="welcome-eyebrow">NOVACART SUPPORT</p>
              <h1>How can I help you today?</h1>
              <p className="welcome-description">
                Ask a question and I’ll find an answer in NovaCart’s knowledge base.
              </p>
              <div className="suggestion-grid" aria-label="Suggested questions">
                {suggestedQuestions.map((question) => (
                  <button
                    className="suggestion-button"
                    key={question}
                    type="button"
                    onClick={() => startTurn(question)}
                  >
                    <span>{question}</span>
                    <svg aria-hidden="true" viewBox="0 0 20 20" fill="none">
                      <path d="M5 10h10m-4-4 4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <ol className="message-list" aria-label="Conversation messages" aria-live="polite">
              {visibleMessages.map((message) => (
                <li key={message.id}>
                  <ChatMessageView message={message} />
                </li>
              ))}
              {loading ? (
                <li className="loading-row">
                  <div className="loading-indicator" role="status" aria-live="polite">
                    <span className="loading-spinner" aria-hidden="true" />
                    <span>
                      {loadingStage === "searching"
                        ? "Searching NovaCart’s knowledge base…"
                        : "Preparing your answer…"}
                    </span>
                  </div>
                </li>
              ) : null}
              {error && pendingTurn ? (
                <li>
                  <div className="error-state" role="alert">
                    <p>Something went wrong. Please try again.</p>
                    <button
                      type="button"
                      onClick={() => void sendTurn(pendingTurn)}
                      disabled={loading}
                    >
                      Retry
                    </button>
                  </div>
                </li>
              ) : null}
            </ol>
          )}
          <div ref={endOfMessages} />
        </section>

        <form className="composer" onSubmit={submitMessage}>
          <label className="sr-only" htmlFor="chat-input">Ask NovaCart a question</label>
          <textarea
            id="chat-input"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleInputKeyDown}
            placeholder="Ask NovaCart anything..."
            rows={2}
            maxLength={4000}
            disabled={inputDisabled}
          />
          <div className="composer-footer">
            <span>Enter to send <span aria-hidden="true">·</span> Shift+Enter for a new line</span>
            <button
              className="send-button"
              type="submit"
              aria-label="Send message"
              disabled={inputDisabled || input.trim().length === 0}
            >
              <svg aria-hidden="true" viewBox="0 0 20 20" fill="none">
                <path d="M4 10h12m-5-5 5 5-5 5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
        </form>
        <p className="privacy-note">Answers are based on NovaCart’s available knowledge base.</p>
      </main>
    </div>
  );
}
