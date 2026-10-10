import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["ROUTER_PATH"] = os.path.join(_tmp, "test_router.joblib")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "test.db")

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["router_trained"] is True


def test_easy_vs_hard_separation(client):
    easy = client.post("/optimize", json={"query": "What is 2 + 2?"}).json()
    hard = client.post("/optimize", json={
        "query": "Design a distributed consensus algorithm for kubernetes and prove it"
    }).json()
    assert easy["recommended_tier"] == "small", \
        f"easy got {easy['recommended_tier']} probs={easy['alternatives']}"
    assert hard["recommended_tier"] == "frontier", \
        f"hard got {hard['recommended_tier']} probs={hard['alternatives']}"


def test_medium_query(client):
    r = client.post("/optimize", json={
        "query": "Explain the difference between REST and GraphQL APIs"
    }).json()
    assert r["recommended_tier"] == "medium", \
        f"medium got {r['recommended_tier']} probs={r['alternatives']}"


def test_feedback_triggers_retrain(client):
    retrained = False
    for i in range(12):
        route = client.post("/optimize",
                            json={"query": f"sanity check {i}"}).json()
        fb = client.post("/feedback", json={
            "request_id": route["request_id"],
            "tier_used": route["recommended_tier"],
            "succeeded": True,
        }).json()
        if fb["router_updated"]:
            retrained = True
    assert retrained, "router never retrained after 12 feedbacks"


def test_stats(client):
    r = client.get("/stats").json()
    assert r["total_routes"] >= 6
    assert r["total_feedback"] >= 12
