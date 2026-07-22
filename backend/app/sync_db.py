"""Синхронный доступ к БД для RQ-задач (worker живёт вне event loop)."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

sync_engine = create_engine(settings.dsn, pool_pre_ping=True)
SyncSession = sessionmaker(sync_engine)
