import logging
import os
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.db import get_db
from app.rbf_router import (
    RBFRouter, TIERS, bootstrap_synthetic_data, extract_features,
)
from app.rlvr import FeedbackSignal, RLVRFeedbackEngine
from app.schemas import (
    FeedbackRequest, FeedbackResponse, HealthResponse,
    RouteRequest, RouteResponse, StatsResponse, ModelTier,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rbf-router")

STATE = {"router": None, "rlvr": None, "last_train_at": 0}
TIER_COST_PER_1M = {"small": 0.15, "medium": 0.60, "frontier": 10.00}
AVG_TOKENS_PER_REQ = 600


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RBF-Router API...")
    router_path = os.getenv("ROUTER_PATH", "router.joblib")
    loaded = None
    if os.path.exists(router_path):
        logger.info("Loading trained router from %s", router_path)
        loaded = RBFRouter.load(router_path)
        if not loaded.is_trained:
            loaded = None
        elif getattr(loaded.scaler, "n_features_in_", None) != len(extract_features("probe")):
            loaded = None
    if loaded is not None:
        STATE["router"] = loaded
    else:
        logger.info("Bootstrapping with synthetic data...")
        X_syn, y_syn = bootstrap_synthetic_data()
        r = RBFRouter(n_centers=9, gamma=0.05)
        r.fit(X_syn, y_syn)
        STATE["router"] = r
        r.save(router_path)
    STATE["rlvr"] = RLVRFeedbackEngine()
    STATE["last_train_at"] = 0
    logger.info("Startup complete.")
    yield


app = FastAPI(title="RBF-Router", version="0.1.0",
              description="Self-improving LLM query router.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def fallback_for(tier: str) -> str:
    order = ["small", "medium", "frontier"]
    return order[min(order.index(tier) + 1, len(order) - 1)]


def expected_savings_pct(tier: str) -> float:
    f = TIER_COST_PER_1M["frontier"] * AVG_TOKENS_PER_REQ / 1_000_000
    c = TIER_COST_PER_1M[tier] * AVG_TOKENS_PER_REQ / 1_000_000
    return float(round((1 - c / f) * 100.0, 2)) if f > 0 else 0.0


def maybe_retrain() -> bool:
    db = get_db()
    total = db.count_feedback()
    if not STATE["rlvr"].should_retrain(total, STATE["last_train_at"]):
        return False
    rows = db.get_all_routes_with_feedback()
    if len(rows) < 3:
        return False
    X, y = [], []
    for row in rows:
        feat = (np.frombuffer(row["embedding"], dtype=np.float64)
                if row["embedding"] else None)
        if feat is None or feat.size == 0:
            continue
        label = row["tier_used"] if row["succeeded"] else fallback_for(row["tier_used"])
        if label not in TIERS:
            continue
        sig = FeedbackSignal(
            request_id=row["request_id"], query_embedding=feat,
            recommended_tier=row["recommended_tier"], tier_used=row["tier_used"],
            succeeded=bool(row["succeeded"]), confidence=float(row["confidence"]),
            quality_score=row["quality_score"])
        vrs = STATE["rlvr"].compute_vrs(sig)
        reps = STATE["rlvr"].repetitions_for(vrs)
        for _ in range(reps):
            X.append(feat)
            y.append(TIERS.index(label))
    if len(X) < 3:
        return False
    router = RBFRouter(n_centers=min(9, max(3, len(X) // 2)), gamma=0.05)
    try:
        router.fit(np.vstack(X), np.array(y, dtype=np.int64))
    except ValueError as e:
        logger.warning("Retrain skipped: %s", e)
        return False
    STATE["router"] = router
    STATE["last_train_at"] = total
    router.save(os.getenv("ROUTER_PATH", "router.joblib"))
    logger.info("Router retrained on %d samples.", len(X))
    return True


@app.get("/ping")
def ping():
    return {"pong": True}


@app.get("/health", response_model=HealthResponse)
def health(response: Response):
    response.headers["Cache-Control"] = "no-store"
    db = get_db()
    return HealthResponse(
        status="ok", version=app.version,
        total_routes=db.count_routes(),
        total_feedback=db.count_feedback(),
        router_trained=bool(STATE["router"] and STATE["router"].is_trained))


def _optimize(req: RouteRequest) -> RouteResponse:
    router = STATE["router"]
    if router is None or not router.is_trained:
        raise HTTPException(503, "Router not ready.")
    feat = extract_features(req.query)
    probs = router.predict_proba(feat.reshape(1, -1))[0]
    idx = int(probs.argmax())
    confidence = float(probs[idx])
    tier = TIERS[idx]
    request_id = str(uuid.uuid4())
    db = get_db()
    db.insert_route(
        request_id=request_id, query=req.query, user_id=req.user_id,
        recommended_tier=tier, confidence=confidence,
        fallback_tier=fallback_for(tier),
        embedding_bytes=feat.astype(np.float64).tobytes())
    return RouteResponse(
        request_id=request_id, recommended_tier=ModelTier(tier),
        confidence=confidence, expected_cost_savings_pct=expected_savings_pct(tier),
        fallback_tier=ModelTier(fallback_for(tier)),
        alternatives={TIERS[i]: float(probs[i]) for i in range(len(TIERS))})


@app.post("/optimize", response_model=RouteResponse)
def optimize(req: RouteRequest, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return _optimize(req)


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return _optimize(req)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest, response: Response):
    response.headers["Cache-Control"] = "no-store"
    db = get_db()
    route_row = db.get_route(req.request_id)
    if route_row is None:
        raise HTTPException(404, f"Unknown request_id: {req.request_id}")
    route_owner = route_row.get("user_id")
    if route_owner and route_owner != req.user_id:
        raise HTTPException(403, "Feedback can only be submitted for your own routes.")
    feat = (np.frombuffer(route_row["embedding"], dtype=np.float64)
            if route_row.get("embedding") else None)
    vrs = 0.0
    if feat is not None and feat.size:
        sig = FeedbackSignal(
            request_id=req.request_id, query_embedding=feat,
            recommended_tier=route_row["recommended_tier"],
            tier_used=req.tier_used.value, succeeded=req.succeeded,
            confidence=float(route_row["confidence"]), quality_score=req.quality_score)
        vrs = STATE["rlvr"].compute_vrs(sig)
    db.insert_feedback(
        request_id=req.request_id, tier_used=req.tier_used.value,
        succeeded=req.succeeded, quality_score=req.quality_score, vrs=vrs)
    updated = maybe_retrain()
    return FeedbackResponse(
        acknowledged=True, total_feedback=db.count_feedback(),
        router_updated=updated,
        message="Feedback recorded. Router retrained." if updated else "Feedback recorded.")


@app.get("/stats", response_model=StatsResponse)
def stats(response: Response):
    response.headers["Cache-Control"] = "no-store"
    db = get_db()
    s = db.get_stats()
    return StatsResponse(
        total_routes=db.count_routes(), total_feedback=db.count_feedback(),
        accuracy_by_tier=s["accuracy_by_tier"],
        avg_confidence=s["avg_confidence"],
        tier_distribution=s["tier_distribution"])


@app.post("/admin/retrain")
def admin_retrain(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return {"retrained": maybe_retrain(),
            "total_feedback": get_db().count_feedback()}
