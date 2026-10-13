import Link from "next/link";
import type { ReactNode } from "react";

type AppHeaderProps = {
  active: "assistant" | "knowledge-base";
  actions?: ReactNode;
};

export default function AppHeader({ active, actions }: AppHeaderProps) {
  return (
    <header className="topbar">
      <Link className="brand" href="/" aria-label="NovaCart AI Customer Assistant">
        <span className="brand-mark" aria-hidden="true">N</span>
        <span className="brand-copy">
          <strong>NovaCart</strong>
          <span>AI Customer Assistant</span>
        </span>
      </Link>

      <div className="header-actions">
        <nav className="app-navigation" aria-label="Main navigation">
          <Link
            className={active === "assistant" ? "app-navigation-link is-active" : "app-navigation-link"}
            href="/"
            aria-current={active === "assistant" ? "page" : undefined}
          >
            AI Assistant
          </Link>
          <Link
            className={active === "knowledge-base" ? "app-navigation-link is-active" : "app-navigation-link"}
            href="/knowledge-base"
            aria-current={active === "knowledge-base" ? "page" : undefined}
          >
            Knowledge Base
          </Link>
        </nav>
        {actions}
      </div>
    </header>
  );
}
