from dataclasses import dataclass
from typing import Optional
import numpy as np

TIER_INDEX = {"small": 0, "medium": 1, "frontier": 2}
ALPHA, BETA, GAMMA, ETA, LAMBDA = 0.4, 0.3, 0.15, 0.3, 0.25
N_MIN, N_MAX, SCALE_K = 1.0, 10.0, 5.0


@dataclass
class FeedbackSignal:
    request_id: str
    query_embedding: np.ndarray
    recommended_tier: str
    tier_used: str
    succeeded: bool
    confidence: float
    quality_score: Optional[float]


class RLVRFeedbackEngine:
    def __init__(self, min_feedback_to_retrain: int = 10,
                 retrain_every_n_new: int = 10):
        self._seen = []
        self.min_feedback_to_retrain = min_feedback_to_retrain
        self.retrain_every_n_new = retrain_every_n_new

    def compute_vrs(self, s: FeedbackSignal) -> float:
        td = 1.0 if s.recommended_tier != s.tier_used else 0.0
        unc = 1.0 - float(np.clip(s.confidence, 0.0, 1.0))
        nov = self._novelty(s.query_embedding)
        rew = 1.0 if s.succeeded else -1.0
        pe = self._prediction_error(s.recommended_tier, s.tier_used, s.succeeded)
        return float(ALPHA * td + BETA * unc + GAMMA * nov + ETA * abs(rew) - LAMBDA * pe)

    def _novelty(self, emb: np.ndarray) -> float:
        if not self._seen:
            self._seen.append(emb)
            return 1.0
        past = np.stack(self._seen[-500:])
        e = emb / (np.linalg.norm(emb) + 1e-9)
        p = past / (np.linalg.norm(past, axis=1, keepdims=True) + 1e-9)
        sims = p @ e
        self._seen.append(emb)
        return float(1.0 - np.max(sims))

    @staticmethod
    def _prediction_error(rec, used, succeeded):
        if rec == used and succeeded:
            return 0.0
        d = abs(TIER_INDEX.get(used, 1) - TIER_INDEX.get(rec, 1)) / 2.0
        return float(min(1.0, d + (0.0 if succeeded else 0.3)))

    @staticmethod
    def repetitions_for(vrs: float) -> int:
        return int(np.clip(SCALE_K * max(vrs, 0.0), N_MIN, N_MAX))

    def should_retrain(self, total: int, last_train_at: int) -> bool:
        if total < self.min_feedback_to_retrain:
            return False
        return (total - last_train_at) >= self.retrain_every_n_new
