from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    FRONTIER = "frontier"


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    user_id: Optional[str] = Field(None, max_length=64)


class RouteResponse(BaseModel):
    request_id: str
    recommended_tier: ModelTier
    confidence: float
    expected_cost_savings_pct: float
    fallback_tier: ModelTier
    alternatives: dict[str, float]
    routed_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackRequest(BaseModel):
    request_id: str
    tier_used: ModelTier
    succeeded: bool
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    user_id: Optional[str] = Field(None, max_length=64)


class FeedbackResponse(BaseModel):
    acknowledged: bool
    total_feedback: int
    router_updated: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    total_routes: int
    total_feedback: int
    router_trained: bool


class StatsResponse(BaseModel):
    total_routes: int
    total_feedback: int
    accuracy_by_tier: dict[str, float]
    avg_confidence: float
    tier_distribution: dict[str, int]
