import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models import INTERACTION_KINDS


class UserCreate(BaseModel):
    name: str | None = Field(default=None, max_length=100)


class UserOut(BaseModel):
    id: uuid.UUID
    name: str | None
    genres: list[str]
    max_price: float | None
    blocked_tags: list[str]

    model_config = {"from_attributes": True}


class PreferencesIn(BaseModel):
    genres: list[str] = Field(default_factory=list, max_length=20)
    max_price: float | None = Field(default=None, ge=0)
    blocked_tags: list[str] = Field(min_length=1, max_length=20)


class GameOut(BaseModel):
    id: int
    title: str
    description: str
    tags: list[str]
    price: float
    positive_ratio: int
    user_reviews: int

    model_config = {"from_attributes": True}


class InteractionIn(BaseModel):
    game_id: int
    kind: str = Field(pattern=f"^({'|'.join(INTERACTION_KINDS)})$")


class InteractionOut(BaseModel):
    game_id: int
    kind: str
    duplicate: bool


class RecommendationItem(BaseModel):
    game: GameOut
    rank: int
    score: float
    reason: str


class RecommendationsOut(BaseModel):
    user_id: uuid.UUID
    model_name: str
    model_version: str
    items: list[RecommendationItem]


class ModelVersionOut(BaseModel):
    id: int
    name: str
    version: str
    params: dict
    metrics: dict
    trained_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}
