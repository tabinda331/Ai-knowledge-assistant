import ChatWorkspace from "@/app/components/chat-workspace";

export default function Home() {
  return (
    <main className="app-shell">
      <ChatWorkspace />
      <footer className="page-footer">
        <span>Information is provided from NovaCart’s available knowledge base.</span>
        <span>AI CUSTOMER ASSISTANT</span>
      </footer>
    </main>
  );
}
