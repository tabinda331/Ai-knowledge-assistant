from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.schemas.ask import AskRequest, AskResponse
from app.schemas.search import SearchRequest
from app.services.answer_generation import generate_grounded_answer
from app.services.llm_provider import (
    LLMConfigurationError,
    LLMServiceError,
    LLMTimeoutError,
    create_llm_provider,
)
from app.services.search_pipeline import SearchPipelineError, run_search
from app.services.conversation_service import (
    create_conversation,
    get_conversation,
    load_history,
    persist_turn,
    resolve_retrieval_query,
)

router = APIRouter(tags=["answers"])


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    db: Session = Depends(get_db_session),
) -> AskResponse:
    if request.conversation_id is None:
        conversation = create_conversation(db)
        history = []
    else:
        conversation = get_conversation(db, request.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        history = load_history(db, conversation.id)

    retrieval_query = resolve_retrieval_query(request.query, history)
    search_request = SearchRequest(
        query=retrieval_query,
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        document_id=request.document_id,
        retrieval_mode=request.retrieval_mode,
        rerank=request.rerank,
    )
    try:
        retrieval = run_search(db, search_request)
    except SearchPipelineError as error:
        db.rollback()
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None

    try:
        if retrieval.results:
            provider = create_llm_provider()
            response = generate_grounded_answer(
                request.query,
                retrieval,
                conversation_id=conversation.id,
                provider=provider,
                history=[{"role": item.role, "content": item.content} for item in history],
            )
        else:
            response = generate_grounded_answer(
                request.query, retrieval, conversation_id=conversation.id
            )
        persist_turn(
            db,
            conversation,
            request.query,
            response.answer,
            [source.model_dump(mode="json") for source in response.sources],
            {
                "retrieval_mode": response.retrieval_mode,
                "reranked": response.reranked,
                "grounded": response.grounded,
                "retrieval_query": retrieval_query,
            },
        )
        return response
    except LLMConfigurationError as error:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(error)) from None
    except LLMTimeoutError as error:
        db.rollback()
        raise HTTPException(status_code=504, detail=str(error)) from None
    except LLMServiceError as error:
        db.rollback()
        raise HTTPException(status_code=502, detail=str(error)) from None
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=503, detail="Conversation service unavailable") from None
