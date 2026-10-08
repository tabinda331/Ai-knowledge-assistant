from __future__ import annotations

from dataclasses import dataclass

from app.services.pdf_extraction import ExtractedPage

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str
    page_start: int
    page_end: int


def _page_spans(pages: list[ExtractedPage]) -> tuple[str, list[tuple[int, int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0

    for page in pages:
        if not page.text:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(page.text)
        cursor += len(page.text)
        spans.append((start, cursor, page.page_number))

    return "".join(parts), spans


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text, page_spans = _page_spans(pages)
    chunks: list[TextChunk] = []
    start = 0

    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            minimum_end = start + max(1, chunk_size // 2)
            boundary = max(text.rfind("\n\n", minimum_end, hard_end), text.rfind(" ", minimum_end, hard_end))
            if boundary > start:
                end = boundary

        content = text[start:end].strip()
        if content:
            covered_pages = [
                page_number
                for page_start, page_end, page_number in page_spans
                if page_start < end and page_end > start
            ]
            chunks.append(
                TextChunk(
                    chunk_index=len(chunks),
                    content=content,
                    page_start=min(covered_pages),
                    page_end=max(covered_pages),
                )
            )

        if end == len(text):
            break
        start = max(start + 1, end - overlap)

    return chunks