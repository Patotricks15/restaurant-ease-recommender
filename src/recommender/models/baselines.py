"""Classical baselines for the restaurant recommendation experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.splitting import Split


def _top_k(scores: np.ndarray, k: int, exclude: set[int] | None) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be positive.")
    ranked = scores.copy()
    if exclude:
        ranked[list(exclude)] = -np.inf
    k = min(k, int(np.isfinite(ranked).sum()))
    if k == 0:
        return np.array([], dtype=np.int64)
    top = np.argpartition(-ranked, k - 1)[:k]
    return top[np.argsort(-ranked[top], kind="stable")]


@dataclass
class PopularityRecommender:
    item_scores: np.ndarray | None = None
    n_items: int = 0

    def fit(self, split: Split) -> PopularityRecommender:
        self.n_items = split.n_items
        self.item_scores = np.bincount(
            split.train["item_index"].to_numpy(dtype=np.int64),
            minlength=self.n_items,
        ).astype(np.float64)
        return self

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None  # noqa: ARG002
    ) -> np.ndarray:
        if self.item_scores is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        return _top_k(self.item_scores, k, exclude)


@dataclass
class RandomRecommender:
    random_state: int = 0
    n_items: int = 0

    def fit(self, split: Split) -> RandomRecommender:
        self.n_items = split.n_items
        return self

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        scores = np.random.default_rng(self.random_state + user_index).random(self.n_items)
        return _top_k(scores, k, exclude)
