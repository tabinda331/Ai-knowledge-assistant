import type { ReactNode } from "react";

import type { ChatSource } from "@/lib/api";

type AnswerContentProps = {
  content: string;
  messageId: string;
  sources: ChatSource[];
};

type TextBlock =
  | { kind: "paragraph"; text: string }
  | { kind: "unordered-list" | "ordered-list"; items: string[] };

export function sourceElementId(messageId: string, sourceNumber: number): string {
  const safeMessageId = messageId.replace(/[^a-zA-Z0-9_-]/g, "");
  return `source-${safeMessageId}-${sourceNumber}`;
}

function renderInline(
  text: string,
  messageId: string,
  sources: ChatSource[],
): ReactNode[] {
  const tokenPattern = /\[SOURCE\s+(\d+)\]|\[(\d+)\]|\*\*([^*]+)\*\*/gi;
  const nodes: ReactNode[] = [];
  let previousIndex = 0;
  let tokenIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = tokenPattern.exec(text)) !== null) {
    if (match.index > previousIndex) {
      nodes.push(text.slice(previousIndex, match.index));
    }

    if (match[3] !== undefined) {
      nodes.push(<strong key={`bold-${tokenIndex}`}>{match[3]}</strong>);
    } else {
      const sourceNumber = Number(match[1] ?? match[2]);
      const hasSource = sourceNumber > 0 && sourceNumber <= sources.length;

      if (hasSource) {
        nodes.push(
          <a
            className="citation-reference"
            href={`#${sourceElementId(messageId, sourceNumber)}`}
            aria-label={`View source ${sourceNumber}`}
            key={`citation-${tokenIndex}`}
          >
            [{sourceNumber}]
          </a>,
        );
      } else if (match[1] !== undefined) {
        // Normalize backend markers even when the corresponding source is absent.
        nodes.push(`[${sourceNumber}]`);
      } else {
        nodes.push(match[0]);
      }
    }

    previousIndex = tokenPattern.lastIndex;
    tokenIndex += 1;
  }

  if (previousIndex < text.length) {
    nodes.push(text.slice(previousIndex));
  }

  return nodes;
}

function parseBlocks(content: string): TextBlock[] {
  const blocks: TextBlock[] = [];
  let paragraph: string[] = [];
  let list: { kind: "unordered-list" | "ordered-list"; items: string[] } | null = null;

  function flushParagraph() {
    if (paragraph.length > 0) {
      blocks.push({ kind: "paragraph", text: paragraph.join(" ") });
      paragraph = [];
    }
  }

  function flushList() {
    if (list) {
      blocks.push(list);
      list = null;
    }
  }

  for (const rawLine of content.replace(/\r\n?/g, "\n").split("\n")) {
    const line = rawLine.trim();
    const unorderedItem = line.match(/^[-*+]\s+(.+)$/);
    const orderedItem = line.match(/^\d+[.)]\s+(.+)$/);

    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }

    if (unorderedItem || orderedItem) {
      flushParagraph();
      const kind = unorderedItem ? "unordered-list" : "ordered-list";
      if (!list || list.kind !== kind) {
        flushList();
        list = { kind, items: [] };
      }
      list.items.push((unorderedItem ?? orderedItem)![1]);
      continue;
    }

    if (list) {
      list.items[list.items.length - 1] += ` ${line}`;
    } else {
      paragraph.push(line);
    }
  }

  flushParagraph();
  flushList();
  return blocks;
}

export default function AnswerContent({ content, messageId, sources }: AnswerContentProps) {
  const blocks = parseBlocks(content);

  return (
    <div className="answer-content">
      {blocks.map((block, blockIndex) => {
        if (block.kind === "paragraph") {
          return (
            <p key={`paragraph-${blockIndex}`}>
              {renderInline(block.text, messageId, sources)}
            </p>
          );
        }

        const ListTag = block.kind === "ordered-list" ? "ol" : "ul";
        return (
          <ListTag key={`${block.kind}-${blockIndex}`}>
            {block.items.map((item, itemIndex) => (
              <li key={`item-${itemIndex}`}>
                {renderInline(item, messageId, sources)}
              </li>
            ))}
          </ListTag>
        );
      })}
    </div>
  );
}
