from redis import Redis
from rq import Queue

from app.config import settings

QUEUE_NAME = "sidequest"


def get_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=Redis.from_url(settings.redis_url))
