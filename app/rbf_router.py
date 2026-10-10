from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
TIERS = ["small", "medium", "frontier"]

_COMPLEXITY_KEYWORDS = (
    "architecture", "algorithm", "proof", "kubernetes", "microservices",
    "design", "optimize", "distributed", "concurrency", "security",
    "analyze", "explain in detail", "compare", "derive", "theorem",
)
_CODE_HINTS = ("def ", "class ", "import ", "function", "```",
               "select ", "for (", "while (")


def extract_features(query: str) -> np.ndarray:
    q = query.strip()
    q_lower = q.lower()
    words = q.split()
    word_count = len(words)
    char_count = len(q)
    avg_word_len = (char_count / word_count) if word_count else 0.0
    digit_count = sum(c.isdigit() for c in q)
    question_marks = q.count("?")
    sentence_count = max(1, len(re.split(r"[.!?]+", q)) - 1)
    keyword_hits = sum(1 for kw in _COMPLEXITY_KEYWORDS if kw in q_lower)
    code_hits = sum(1 for hint in _CODE_HINTS if hint in q_lower)
    unique_ratio = (len(set(words)) / word_count) if word_count else 0.0
    return np.array([
        char_count, word_count, avg_word_len,
        digit_count, question_marks, sentence_count,
        keyword_hits, code_hits, unique_ratio,
    ], dtype=np.float64)


def _as_feature_matrix(X) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, str):
        X = [X]
    return np.stack([extract_features(q) for q in X])


def bootstrap_synthetic_data():
    queries = [
        "What is 2 + 2?",
        "Write a quick python print statement",
        "What's the capital of France?",
        "Convert 10 miles to kilometers",
        "How do I reverse a string in python?",
        "What year did WW2 end?",
        "Summarize this paragraph briefly",
        "Write unit tests for this function",
        "Explain the difference between REST and GraphQL",
        "Refactor this function to be more readable",
        "Write a SQL query to join two tables",
        "Explain how a hash map works",
        "Explain quantum field theory and write a mathematical proof",
        "Design an enterprise microservices architecture with kubernetes",
        "Derive the closed-form solution for a distributed consensus protocol",
        "Compare and analyze the security trade-offs of three concurrency models",
        "Write a formal proof of correctness for this distributed algorithm",
        "Design a fault-tolerant horizontally scalable event-sourcing architecture",
    ]
    labels = [0] * 6 + [1] * 6 + [2] * 6
    return queries, np.array(labels, dtype=np.int64)


class RBFRouter:
    def __init__(self, n_centers: int = 9, gamma: float = 0.05):
        self.n_centers = n_centers
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.centers: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.is_trained = False

    def _rbf(self, X, C):
        dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        return np.exp(-self.gamma * (dists ** 2))

    def fit(self, X, y):
        features = _as_feature_matrix(X)
        n_samples = features.shape[0]
        n_centers = min(self.n_centers, n_samples)
        if n_centers < 1:
            return self
        self.n_centers = n_centers
        y_onehot = np.zeros((n_samples, len(TIERS)))
        y_onehot[np.arange(n_samples), y] = 1.0
        X_scaled = self.scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=self.n_centers, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        self.centers = kmeans.cluster_centers_
        H = self._rbf(X_scaled, self.centers)
        self.weights = np.linalg.pinv(H) @ y_onehot
        self.is_trained = True
        logger.info("RBF router trained: %d samples, %d centers",
                    n_samples, self.n_centers)
        return self

    def predict_proba(self, X) -> np.ndarray:
        features = _as_feature_matrix(X)
        if not self.is_trained:
            return np.full((features.shape[0], len(TIERS)), 1.0 / len(TIERS))
        X_scaled = self.scaler.transform(features)
        H = self._rbf(X_scaled, self.centers)
        raw = H @ self.weights
        raw = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(raw)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, query: str):
        if not self.is_trained:
            return 0, 0.50, {"small": 0.33, "medium": 0.33, "frontier": 0.34}
        probs = self.predict_proba(query)[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]), {
            TIERS[i]: float(probs[i]) for i in range(len(TIERS))
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "n_centers": self.n_centers, "gamma": self.gamma,
            "scaler": self.scaler, "centers": self.centers,
            "weights": self.weights, "is_trained": self.is_trained,
        }, path)

    @classmethod
    def load(cls, path: str):
        state = joblib.load(path)
        router = cls(n_centers=state["n_centers"], gamma=state["gamma"])
        router.scaler = state["scaler"]
        router.centers = state["centers"]
        router.weights = state["weights"]
        router.is_trained = state["is_trained"]
        return router
