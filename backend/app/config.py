import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def get_database_url() -> str | None:
    return os.getenv("DATABASE_URL")


def get_document_storage_root() -> Path:
    configured_path = os.getenv("DOCUMENT_STORAGE_DIR")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "storage"


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def get_reranker_model_name() -> str:
    return os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def get_llm_api_key() -> str | None:
    return os.getenv("LLM_API_KEY")


def get_llm_model() -> str:
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


def get_llm_base_url() -> str | None:
    return os.getenv("LLM_BASE_URL") or None


def get_conversation_history_messages() -> int:
    try:
        value = int(os.getenv("CONVERSATION_HISTORY_MESSAGES", "10"))
    except ValueError:
        return 10
    return max(0, min(value, 100))
