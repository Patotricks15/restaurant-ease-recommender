"""Shared fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import build_implicit_feedback, user_holdout_split  # noqa: E402
from recommender.data.loader import InteractionData  # noqa: E402


@pytest.fixture
def interactions() -> InteractionData:
    rng = np.random.default_rng(0)
    rows = []
    for user_id in range(1, 41):
        n_visits = rng.integers(4, 12)
        places = rng.choice(30, size=n_visits, replace=False)
        for place in places:
            rows.append({"user_id": user_id, "place_id": int(place) + 1, "rating": 2})
    ratings = pd.DataFrame(rows)
    return build_implicit_feedback(ratings, min_user_interactions=2, min_item_interactions=2)


@pytest.fixture
def split(interactions: InteractionData):
    return user_holdout_split(interactions, random_state=0)


@pytest.fixture
def catalogue(interactions: InteractionData) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "place_id": interactions.item_ids,
            "price": rng.choice(["low", "medium", "high"], size=interactions.n_items),
            "cuisine": rng.choice(["Mexican", "Bar", "Cafeteria"], size=interactions.n_items),
        }
    )
