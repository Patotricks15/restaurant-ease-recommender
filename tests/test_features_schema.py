"""Schema-contract tests for the feature spec and the raw data loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recommender.data.loader import build_implicit_feedback, load_ratings
from recommender.features.spec import load_feature_spec

ROOT = Path(__file__).resolve().parents[1]


def test_feature_spec_declares_known_families_only():
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    assert spec.names() == {
        "user_activity_count",
        "user_mean_rating",
        "item_popularity_rank",
        "item_price_tier",
        "item_cuisine_tokens",
    }


def test_load_ratings_rejects_missing_columns(tmp_path):
    bad = tmp_path / "ratings.csv"
    pd.DataFrame({"user_id": [1], "rating": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_ratings(bad)


def test_load_ratings_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ratings(tmp_path / "missing.csv")


def test_build_implicit_feedback_drops_non_positive_ratings():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "place_id": [10, 11, 12, 10, 11, 13],
            "rating": [0, 1, 2, 1, 1, 0],
        }
    )
    data = build_implicit_feedback(ratings, min_user_interactions=1, min_item_interactions=1)
    assert (data.frame["rating"] >= 1).all()
