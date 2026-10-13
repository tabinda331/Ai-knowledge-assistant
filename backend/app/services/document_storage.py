from __future__ import annotations

import hashlib
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.config import get_document_storage_root

MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024
PDF_SIGNATURE = b"%PDF-"
_UNSAFE_FILENAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._ -]+")


class InvalidPDFError(ValueError):
    pass


class PDFTooLargeError(ValueError):
    pass


class DocumentStorageError(OSError):
    pass


@dataclass(frozen=True)
class StoredPDF:
    storage_key: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class StagedPDFRemoval:
    original_path: Path
    staged_path: Path


def sanitize_filename(filename: str | None) -> str:
    if not filename:
        raise InvalidPDFError("A PDF filename is required")

    basename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    sanitized = _UNSAFE_FILENAME_CHARACTERS.sub("_", basename).strip(" .")
    sanitized = sanitized[:240].strip(" .")
    if not sanitized or Path(sanitized).suffix.lower() != ".pdf":
        raise InvalidPDFError("Only PDF files are accepted")
    return sanitized


def validate_pdf_content_type(content_type: str | None) -> None:
    if content_type and content_type.lower() not in {
        "application/pdf",
        "application/octet-stream",
    }:
        raise InvalidPDFError("Only PDF files are accepted")


def store_pdf(file_stream: BinaryIO, document_id: uuid.UUID) -> StoredPDF:
    storage_root = get_document_storage_root()
    relative_directory = Path("documents") / str(document_id).replace("-", "")[:2]
    destination_directory = storage_root / relative_directory
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    size_bytes = 0
    header = bytearray()

    try:
        destination_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination_directory, prefix=".upload-", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            while chunk := file_stream.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > MAX_PDF_SIZE_BYTES:
                    raise PDFTooLargeError("PDF exceeds the 25 MiB upload limit")
                if len(header) < 1024:
                    header.extend(chunk[: 1024 - len(header)])
                digest.update(chunk)
                temporary_file.write(chunk)

        if PDF_SIGNATURE not in header:
            raise InvalidPDFError("The uploaded file does not have a PDF signature")

        generated_filename = f"{document_id}.pdf"
        storage_key = (relative_directory / generated_filename).as_posix()
        destination = storage_root / Path(storage_key)
        os.replace(temporary_path, destination)
        temporary_path = None
        return StoredPDF(storage_key, size_bytes, digest.hexdigest())
    except (InvalidPDFError, PDFTooLargeError):
        raise
    except OSError:
        raise DocumentStorageError("Unable to store uploaded document") from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def remove_stored_pdf(storage_key: str) -> None:
    stored_path = resolve_stored_pdf_path(storage_key)
    if stored_path is not None:
        stored_path.unlink(missing_ok=True)


def stage_stored_pdf_removal(storage_key: str | None) -> StagedPDFRemoval | None:
    if not storage_key:
        return None
    if not isinstance(storage_key, str):
        raise DocumentStorageError("Unable to remove stored document")
    try:
        stored_path = resolve_stored_pdf_path(storage_key)
    except (OSError, RuntimeError):
        raise DocumentStorageError("Unable to remove stored document") from None
    if stored_path is None:
        raise DocumentStorageError("Unable to remove stored document")
    if not stored_path.exists():
        return None
    if not stored_path.is_file():
        raise DocumentStorageError("Unable to remove stored document")
    staged_path = stored_path.with_name(f".delete-{uuid.uuid4().hex}.tmp")
    try:
        os.replace(stored_path, staged_path)
    except OSError:
        raise DocumentStorageError("Unable to remove stored document") from None
    return StagedPDFRemoval(stored_path, staged_path)


def restore_staged_pdf(removal: StagedPDFRemoval | None) -> None:
    if removal is None or not removal.staged_path.exists():
        return
    try:
        os.replace(removal.staged_path, removal.original_path)
    except OSError:
        raise DocumentStorageError("Unable to restore stored document") from None


def finalize_staged_pdf_removal(removal: StagedPDFRemoval | None) -> None:
    if removal is None:
        return
    try:
        removal.staged_path.unlink(missing_ok=True)
    except OSError:
        raise DocumentStorageError("Unable to remove stored document") from None


def resolve_stored_pdf_path(storage_key: str) -> Path | None:
    storage_root = get_document_storage_root().resolve()
    stored_path = (storage_root / Path(storage_key)).resolve()
    if not stored_path.is_relative_to(storage_root):
        return None
    return stored_path
