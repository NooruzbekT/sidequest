import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 2},
)

redis_client = redis.Redis.from_url(
    settings.redis_url,
    socket_connect_timeout=2,
    socket_timeout=2,
)


async def ping_db() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def ping_redis() -> bool:
    try:
        return bool(await redis_client.ping())
    except Exception:
        return False
