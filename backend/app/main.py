import logging
import time
import uuid

from fastapi import FastAPI, Request

from app.config import settings
from app.db import ping_db, ping_redis
from app.routers import admin, games, recommendations, users

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sidequest")

app = FastAPI(title="SideQuest API", version=settings.app_version)
app.include_router(users.router)
app.include_router(games.router)
app.include_router(recommendations.router)
app.include_router(admin.router)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health() -> dict:
    db_ok = await ping_db()
    redis_ok = await ping_redis()
    return {
        "status": "ok" if db_ok and redis_ok else "degraded",
        "db": db_ok,
        "redis": redis_ok,
        "version": settings.app_version,
    }
