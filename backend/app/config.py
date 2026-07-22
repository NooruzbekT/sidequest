from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # приоритетнее компонентов, если задан явно
    database_url: str | None = None

    postgres_user: str = "sidequest"
    postgres_password: str = "sidequest"
    postgres_db: str = "sidequest"
    postgres_host: str = "localhost"
    postgres_port: int = 5433

    redis_url: str = "redis://localhost:6379/0"
    model_artifact_path: str = "ml/artifacts/hybrid_v2.json"
    app_version: str = "0.2.0"

    @property
    def artifact_path(self) -> str:
        # относительный путь трактуем от корня репозитория, а не от CWD процесса
        p = Path(self.model_artifact_path)
        return str(p if p.is_absolute() else REPO_ROOT / p)

    @property
    def dsn(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
