"""Top-N ranking metrics for the restaurant recommender."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from recommender.data.splitting import Split


class Recommender(Protocol):
    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray: ...


def _ndcg_at_k(ranked: np.ndarray, targets: set[int], k: int) -> float:
    gains = sum(
        1.0 / np.log2(position + 2)
        for position, item in enumerate(ranked)
        if int(item) in targets
    )
    ideal = sum(1.0 / np.log2(position + 2) for position in range(min(len(targets), k)))
    return gains / ideal if ideal else 0.0


def _average_precision_at_k(ranked: np.ndarray, targets: set[int], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for position, item in enumerate(ranked):
        if int(item) in targets:
            hits += 1
            precision_sum += hits / (position + 1)
    denominator = min(len(targets), k)
    return precision_sum / denominator if denominator else 0.0


@dataclass(frozen=True)
class EvaluationReport:
    model: str
    protocol: str
    k: int
    ndcg: float
    recall: float
    precision: float
    average_precision: float
    hit_rate: float
    coverage: float
    n_users: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "protocol": self.protocol,
            "k": self.k,
            "ndcg": round(self.ndcg, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "map": round(self.average_precision, 4),
            "hit_rate": round(self.hit_rate, 4),
            "coverage": round(self.coverage, 4),
            "n_users": self.n_users,
        }


def evaluate(
    model: Recommender,
    split: Split,
    k: int = 10,
    partition: str = "test",
    model_name: str = "model",
) -> EvaluationReport:
    """Evaluate ranking quality against each user's held-out positives."""
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be 'validation' or 'test'.")
    frame = getattr(split, partition)
    if frame.empty:
        raise ValueError(f"Partition '{partition}' is empty.")

    histories = split.user_positives()
    relevant = {
        int(user): set(group["item_index"].astype(int))
        for user, group in frame.groupby("user_index")
    }
    ndcg_total = recall_total = precision_total = ap_total = 0.0
    hits = 0
    recommended: set[int] = set()

    for user, targets in relevant.items():
        ranked = model.recommend(user, k=k, exclude=histories.get(user, set()))
        recommended.update(map(int, ranked))
        positions = [position for position, item in enumerate(ranked) if int(item) in targets]
        if positions:
            hits += 1
        ndcg_total += _ndcg_at_k(ranked, targets, k)
        ap_total += _average_precision_at_k(ranked, targets, k)
        recall_total += len(positions) / len(targets)
        precision_total += len(positions) / k

    n_users = len(relevant)
    return EvaluationReport(
        model=model_name,
        protocol=split.protocol,
        k=k,
        ndcg=ndcg_total / n_users,
        recall=recall_total / n_users,
        precision=precision_total / n_users,
        average_precision=ap_total / n_users,
        hit_rate=hits / n_users,
        coverage=len(recommended) / split.n_items,
        n_users=n_users,
    )
