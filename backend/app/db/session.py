from functools import lru_cache
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.config import get_database_url


@lru_cache(maxsize=1)
def _create_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def get_engine() -> Engine | None:
    database_url = get_database_url()
    if not database_url:
        return None
    return _create_engine(database_url)


def get_db_session() -> Iterator[Session]:
    engine = get_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured")
    with Session(engine) as session:
        yield session