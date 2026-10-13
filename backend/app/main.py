from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes.documents import router as documents_router
from app.api.routes.ask import router as ask_router
from app.api.routes.search import router as search_router
from app.db.session import get_engine

app = FastAPI(title="AI Knowledge Assistant API")
LOCAL_FRONTEND_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(ask_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/db")
def database_health_check() -> dict[str, str]:
    try:
        engine = get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="Database unavailable")
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable") from None

    return {"status": "ok", "database": "connected"}
