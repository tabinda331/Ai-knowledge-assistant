from __future__ import annotations

from sqlalchemy.orm import Session

from app.schemas.search import SearchRequest, SearchResponse
from app.services.bm25_retrieval import BM25RetrievalError, retrieve_bm25_chunks
from app.services.hybrid_retrieval import fuse_ranked_results
from app.services.reranking import RerankingError, rerank_candidates
from app.services.vector_retrieval import VectorRetrievalError, retrieve_chunks


class SearchPipelineError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def run_search(db: Session, request: SearchRequest) -> SearchResponse:
    retrieval_limit = request.candidate_k if request.rerank else request.top_k
    try:
        if request.retrieval_mode == "dense":
            results = retrieve_chunks(
                db,
                query=request.query,
                top_k=retrieval_limit,
                document_id=request.document_id,
            )
        elif request.retrieval_mode == "bm25":
            results = retrieve_bm25_chunks(
                db,
                query=request.query,
                top_k=retrieval_limit,
                document_id=request.document_id,
            )
        else:
            dense_results = retrieve_chunks(
                db,
                query=request.query,
                top_k=retrieval_limit,
                document_id=request.document_id,
            )
            bm25_results = retrieve_bm25_chunks(
                db,
                query=request.query,
                top_k=retrieval_limit,
                document_id=request.document_id,
            )
            results = fuse_ranked_results(dense_results, bm25_results, retrieval_limit)

        if request.rerank:
            results = rerank_candidates(request.query, results)[: request.top_k]
        else:
            results = results[: request.top_k]
    except VectorRetrievalError as error:
        raise SearchPipelineError(str(error), error.status_code) from None
    except BM25RetrievalError as error:
        raise SearchPipelineError(str(error), error.status_code) from None
    except RerankingError:
        raise SearchPipelineError("Reranking is unavailable", 503) from None

    return SearchResponse(
        query=request.query,
        retrieval_mode=request.retrieval_mode,
        reranked=request.rerank,
        results=results,
    )