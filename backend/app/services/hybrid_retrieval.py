from __future__ import annotations

from app.schemas.search import SearchResult

DEFAULT_RRF_CONSTANT = 60


def fuse_ranked_results(
    dense_results: list[SearchResult],
    bm25_results: list[SearchResult],
    top_k: int = 5,
    rrf_constant: int = DEFAULT_RRF_CONSTANT,
) -> list[SearchResult]:
    candidates: dict[str, dict[str, object]] = {}

    for rank, result in enumerate(dense_results, start=1):
        entry = candidates.setdefault(str(result.chunk_id), {"result": result})
        entry["dense_rank"] = rank
        entry["similarity"] = result.similarity

    for rank, result in enumerate(bm25_results, start=1):
        entry = candidates.setdefault(str(result.chunk_id), {"result": result})
        entry["bm25_rank"] = rank
        entry["bm25_score"] = result.bm25_score

    fused: list[SearchResult] = []
    for entry in candidates.values():
        dense_rank = entry.get("dense_rank")
        bm25_rank = entry.get("bm25_rank")
        score = 0.0
        if dense_rank is not None:
            score += 1.0 / (rrf_constant + int(dense_rank))
        if bm25_rank is not None:
            score += 1.0 / (rrf_constant + int(bm25_rank))
        result = entry["result"]
        assert isinstance(result, SearchResult)
        fused.append(
            result.model_copy(
                update={
                    "similarity": entry.get("similarity"),
                    "bm25_score": entry.get("bm25_score"),
                    "dense_rank": dense_rank,
                    "bm25_rank": bm25_rank,
                    "hybrid_score": score,
                }
            )
        )

    fused.sort(key=lambda result: (-float(result.hybrid_score or 0), result.chunk_index))
    return fused[:top_k]