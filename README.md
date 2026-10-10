# RBF-Router API

A self-improving query router that classifies natural language prompts into three
LLM cost tiers — `small`, `medium`, `frontier` — using a tiny Radial Basis
Function network. Learns from feedback via a custom RLVR loop.

**No LLM. No GPU. No model downloads. ~1 ms per request.**

---

## Why This Exists

Every LLM application faces the same cost problem: send everything to a frontier
model (expensive) or everything to a small model (fails on hard queries). Existing
routers use static heuristics that never improve.

RBF-Router is a lightweight alternative:
- **Runs on CPU** — no GPU, no PyTorch, no transformers
- **~50 KB model** — not hundreds of megabytes
- **Trains in milliseconds** — closed-form solution, no backpropagation
- **Learns from feedback** — adapts to your real usage patterns over time

---

## How It Works

```
User query
    |
    v
[extract_features] --> 9-dim numeric vector
    |
    v
[RBFRouter] --> K-Means centers + pseudo-inverse weights
    |
    v
softmax over 3 tiers
    |
    v
{small: 0.7, medium: 0.2, frontier: 0.1}
    |
    v
Recommend "small", confidence 0.7
    |
    v
Store route in SQLite
    |
    v
User sends /feedback --> RLVR computes VRS --> weighted retrain
```

---

## The Two Core Ideas

### RBF-MoE (Radial Basis Function Mixture of Experts)

Standard Mixture of Experts uses a learned softmax gate — a large weight matrix.
This project uses distance to prototypes instead:

```
gate_weights = softmax(-gamma * ||x - c||^2)
```

where `c` are K-Means centers found during training. Result: ~50 KB instead of
hundreds of MB.

### RLVR (Reinforcement Learning through Virtual Repetition)

Instead of replaying every feedback sample equally, each experience is weighted
by its **learning value**:

```
VRS = alpha * |TD-error|
    + beta  * uncertainty
    + gamma * novelty
    + eta   * |reward|
    - lambda * prediction_error
```

Surprising feedback (confident-but-wrong) gets repeated more during retraining.
Obvious feedback (confident-and-right) gets repeated less. This makes the router
learn faster from its mistakes.

---

## Quick Start

### Prerequisites

- Python 3.11
- pip

### Install

```bash
git clone https://github.com/YOUR-USERNAME/rbf-router-api.git
cd rbf-router-api
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
uvicorn app.main:app --reload --port 8000
```

First startup bootstraps the router from 18 synthetic examples and saves it to
`router.joblib`. Subsequent startups load the trained model.

Open http://localhost:8000/docs for the interactive Swagger UI.

### Test

```bash
pytest tests/ -v
```

**Expected:** 5 passed.

---

## API Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/ping` | No | Liveness probe |
| GET | `/health` | No | Counters, version, router status |
| POST | `/optimize` | Optional | Route a query |
| POST | `/route` | Optional | Alias for `/optimize` |
| POST | `/feedback` | Optional | Report outcome, trigger retraining |
| GET | `/stats` | Optional | Accuracy, distribution, confidence |
| POST | `/admin/retrain` | Optional | Force a retrain (dev) |

### Example: Route a Query

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2 + 2?"}'
```

Response:

```json
{
  "request_id": "e6406b8e-029f-4a38-9d83-e286b901856b",
  "recommended_tier": "small",
  "confidence": 0.55,
  "expected_cost_savings_pct": 98.5,
  "fallback_tier": "medium",
  "alternatives": {
    "small": 0.55,
    "medium": 0.22,
    "frontier": 0.23
  }
}
```

### Example: Send Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "e6406b8e-029f-4a38-9d83-e286b901856b",
    "tier_used": "small",
    "succeeded": true
  }'
```

Response:

```json
{
  "acknowledged": true,
  "total_feedback": 1,
  "router_updated": false,
  "message": "Feedback recorded."
}
```

---

## Architecture

```
rbf-router-api/
├── app/
│   ├── __init__.py
│   ├── main.py          FastAPI app + 7 endpoints
│   ├── rbf_router.py    RBF network, K-Means, feature extraction
│   ├── rlvr.py          VRS computation, adaptive repetition
│   ├── db.py            SQLite layer (routes, feedback)
│   └── schemas.py       Pydantic request/response models
├── tests/
│   ├── __init__.py
│   └── test_api.py      5 regression tests
├── Dockerfile           Lean 180 MB image
├── requirements.txt     Minimal dependencies
└── README.md
```

---

## The Feature Vector

Queries are converted to a 9-dimensional vector. No embeddings, no LLM, no
model download.

| Feature | What it captures |
|---|---|
| `char_count` | Overall length |
| `word_count` | Lexical density |
| `avg_word_len` | Formality |
| `digit_count` | Numeric content |
| `question_marks` | Interrogative form |
| `sentence_count` | Multi-part questions |
| `keyword_hits` | Complexity indicators (architecture, proof, kubernetes...) |
| `code_hits` | Code markers (`def`, `class`, `import`, `SELECT`...) |
| `unique_ratio` | Vocabulary diversity |

---

## Performance

| Metric | Value |
|---|---|
| Cold start | Instant (no model download) |
| Per-request latency | ~1 ms |
| Model size | ~50 KB |
| Docker image | ~180 MB |
| Training time | Milliseconds (closed-form) |
| Bootstrap accuracy | ~80% on 50 hand-labeled queries |

### Known Limitations

- English-only keyword patterns
- Relies on 18 bootstrap examples until real feedback accumulates
- SQLite is single-writer (fine for demo; Postgres for scale)
- Confidence hovers around 0.5 until more diverse feedback arrives

---

## Docker

Build:

```bash
docker build -t rbf-router .
```

Run:

```bash
docker run -p 8000:8000 rbf-router
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `ROUTER_PATH` | `router.joblib` | Where the trained model is saved |
| `DATABASE_PATH` | `rbf_router.db` | SQLite file location |
| `APP_NAME` | `RBF-Router` | API title |
| `APP_VERSION` | `0.1.0` | API version |

---

## Testing

```bash
pytest tests/ -v
```

The suite covers:

1. **`test_health`** — server boots, router is trained
2. **`test_easy_vs_hard_separation`** — easy queries → `small`, hard → `frontier`
3. **`test_medium_query`** — medium queries land in the middle tier
4. **`test_feedback_triggers_retrain`** — RLVR loop fires after threshold
5. **`test_stats`** — counters reflect activity

---

## Design Decisions

**Why RBF instead of a neural network?**
Closed-form training via pseudo-inverse. No epochs, no learning rate, no GPU.
Trains in milliseconds. Perfect for a resource-constrained environment.

**Why 9 features instead of sentence embeddings?**
Embeddings require downloading a model (~90 MB) and running inference per request.
For a router with 3 output classes, handcrafted features are sufficient and run
in microseconds.

**Why SQLite?**
Zero configuration. Single file. Good enough for a demo. Swap for Postgres when
concurrent writes become a bottleneck.

**Why RLVR instead of standard experience replay?**
Standard replay weights every sample equally. RLVR weights samples by learning
value — surprising outcomes get more attention during retraining. Faster
convergence on small datasets.

---

## Roadmap

Completed:
- [x] RBF router with closed-form training
- [x] RLVR feedback loop
- [x] SQLite persistence
- [x] 5 regression tests
- [x] Docker packaging

Future:
- [ ] API key authentication
- [ ] RFC 9457 Problem Details error responses
- [ ] Rate limiting
- [ ] Postgres migration
- [ ] Prometheus metrics
- [ ] CI/CD via GitHub Actions

---

## License

MIT

---

## Acknowledgments

Inspired by research on Mixture of Experts, Radial Basis Function networks, and
experience replay in reinforcement learning. The RLVR formulation is original to
this project.
