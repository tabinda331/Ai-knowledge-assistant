from __future__ import annotations

import uuid

from app.schemas.ask import AskResponse
from app.schemas.search import SearchResponse
from app.services.context_builder import build_context
from app.services.llm_provider import LLMProvider

NO_EVIDENCE_ANSWER = "I couldn't find enough information in the available documents to answer this question."


def generate_grounded_answer(
    question: str,
    retrieval: SearchResponse,
    *,
    conversation_id: uuid.UUID,
    provider: LLMProvider | None = None,
    history: list[dict[str, str]] | None = None,
) -> AskResponse:
    if not retrieval.results:
        return AskResponse(
            conversation_id=conversation_id,
            question=question,
            answer=NO_EVIDENCE_ANSWER,
            sources=[],
            grounded=False,
            retrieval_mode=retrieval.retrieval_mode,
            reranked=retrieval.reranked,
        )

    context = build_context(retrieval.results)
    if provider is None:
        from app.services.llm_provider import create_llm_provider

        provider = create_llm_provider()
    if history:
        answer = provider.generate_answer(context, question, history)
    else:
        answer = provider.generate_answer(context, question)
    return AskResponse(
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        sources=context.sources,
        grounded=True,
        retrieval_mode=retrieval.retrieval_mode,
        reranked=retrieval.reranked,
    )
