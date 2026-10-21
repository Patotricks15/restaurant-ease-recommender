"""Tests for the closed-form EASE^R model and the baselines."""

from __future__ import annotations

import numpy as np

from recommender.evaluation import evaluate
from recommender.models import EASERecommender, PopularityRecommender, RandomRecommender


def test_ease_fit_produces_zero_diagonal(split):
    model = EASERecommender(l2_reg=100.0).fit(split)
    assert model.B is not None
    np.testing.assert_allclose(np.diag(model.B), 0.0)


def test_ease_recommend_excludes_history(split):
    model = EASERecommender(l2_reg=100.0).fit(split)
    history = split.user_positives()[0]
    ranked = model.recommend(0, k=5, exclude=history)
    assert not set(map(int, ranked)) & history


def test_ease_beats_random_on_ndcg(split):
    ease = EASERecommender(l2_reg=100.0).fit(split)
    random_model = RandomRecommender(random_state=0).fit(split)

    ease_report = evaluate(ease, split, k=10, partition="test", model_name="EASE")
    random_report = evaluate(random_model, split, k=10, partition="test", model_name="Random")

    assert ease_report.ndcg >= random_report.ndcg


def test_popularity_recommends_most_frequent_items(split):
    model = PopularityRecommender().fit(split)
    counts = split.train["item_index"].value_counts()
    top_item = int(counts.idxmax())
    ranked = model.recommend(0, k=1, exclude=set())
    assert int(ranked[0]) == top_item


def test_ease_rejects_non_positive_l2_reg(split):
    import pytest

    with pytest.raises(ValueError):
        EASERecommender(l2_reg=0.0).fit(split)
