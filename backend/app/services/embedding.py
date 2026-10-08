from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from app.config import get_embedding_model_name

EMBEDDING_DIMENSION = 1024
EMBEDDING_BATCH_SIZE = 32


class EmbeddingGenerationError(Exception):
    pass


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []

    try:
        model = _load_model(get_embedding_model_name())
        encoded = model.encode(
            list(texts),
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = encoded.tolist()
        if len(vectors) != len(texts) or any(
            len(vector) != EMBEDDING_DIMENSION for vector in vectors
        ):
            raise EmbeddingGenerationError("Embedding model returned an unexpected vector shape")
        return [[float(value) for value in vector] for vector in vectors]
    except EmbeddingGenerationError:
        raise
    except Exception:
        raise EmbeddingGenerationError("Embedding generation failed") from None