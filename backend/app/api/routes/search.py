from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_pipeline import SearchPipelineError, run_search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse, response_model_exclude_none=True)
def search(
    request: SearchRequest,
    db: Session = Depends(get_db_session),
) -> SearchResponse:
    try:
        return run_search(db, request)
    except SearchPipelineError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None