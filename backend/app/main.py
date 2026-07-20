from fastapi import FastAPI

from app.config import settings
from app.db import ping_db, ping_redis

app = FastAPI(title="SideQuest API", version=settings.app_version)


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
