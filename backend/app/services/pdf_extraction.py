from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFExtractionError(ValueError):
    """Raised when a PDF cannot provide usable text without OCR."""


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


def normalize_text(text: str) -> str:
    normalized_lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    current_lines: list[str] = []

    for line in normalized_lines:
        if not line:
            if current_lines:
                paragraphs.append(" ".join(current_lines))
                current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph).strip()


def extract_pdf_pages(pdf_path: Path) -> list[ExtractedPage]:
    try:
        reader = PdfReader(str(pdf_path), strict=False)
        if reader.is_encrypted:
            raise PDFExtractionError("The PDF is encrypted and cannot be processed")
        pages = [
            ExtractedPage(page_number=index, text=normalize_text(page.extract_text() or ""))
            for index, page in enumerate(reader.pages, start=1)
        ]
    except PDFExtractionError:
        raise
    except (OSError, PdfReadError, ValueError, TypeError):
        raise PDFExtractionError("The PDF could not be read") from None

    if not pages or not any(page.text for page in pages):
        raise PDFExtractionError("No usable text could be extracted from the PDF")
    return pages