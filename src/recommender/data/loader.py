"""Load the Restaurant Consumer Ratings data and build implicit feedback.

The dataset (UCI / Kaggle "Restaurant Data with Consumer Ratings") has no
event timestamp, so — unlike ``movielens-svd-ranking`` or
``foodcom-nmf-recommender`` — this project cannot use a temporal split and
instead evaluates with a user-aware random holdout (see ``splitting.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_RATING_COLUMNS = {"user_id", "place_id", "rating"}
MIN_POSITIVE_RATING = 1  # overall rating scale is 0, 1, 2


@dataclass(frozen=True)
class InteractionData:
    """Implicit-feedback interactions with contiguous internal indices.

    Attributes:
        frame: Columns ``user_index``, ``item_index``, ``rating``.
        user_ids: Original user ids, positionally aligned to ``user_index``.
        item_ids: Original place ids, positionally aligned to ``item_index``.
    """

    frame: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return int(self.user_ids.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_ids.shape[0])

    def __len__(self) -> int:
        return int(self.frame.shape[0])


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Read ratings.csv and validate its contract.

    Args:
        path: Path to a ``user_id, place_id, rating`` CSV.

    Returns:
        The raw frame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py --synthetic"
        )
    frame = pd.read_csv(path)
    missing = REQUIRED_RATING_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows.")
    return frame


def build_implicit_feedback(
    ratings: pd.DataFrame,
    min_user_interactions: int = 3,
    min_item_interactions: int = 3,
) -> InteractionData:
    """Binarize ratings, apply an iterative k-core filter and reindex.

    A rating of 1 or 2 (the dataset's "satisfactory"/"highly satisfactory"
    overall rating, on a 0-2 scale) is a positive. A rating of 0 is dropped
    rather than treated as a negative, consistent with the implicit-feedback
    convention used across this portfolio.

    Args:
        ratings: Raw ``user_id, place_id, rating`` frame.
        min_user_interactions: k-core threshold on the user side.
        min_item_interactions: k-core threshold on the item (restaurant) side.

    Returns:
        Interactions with contiguous indices.

    Raises:
        ValueError: If the thresholds eliminate every interaction.
    """
    frame = ratings.loc[ratings["rating"] >= MIN_POSITIVE_RATING].copy()
    frame = frame.drop_duplicates(subset=["user_id", "place_id"], keep="first")

    while True:
        user_counts = frame["user_id"].value_counts()
        item_counts = frame["place_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        keep_items = item_counts[item_counts >= min_item_interactions].index

        filtered = frame[frame["user_id"].isin(keep_users) & frame["place_id"].isin(keep_items)]
        if len(filtered) == len(frame):
            frame = filtered
            break
        frame = filtered
        if frame.empty:
            break

    if frame.empty:
        raise ValueError(
            "Core filtering removed every interaction; thresholds "
            f"({min_user_interactions}, {min_item_interactions}) are too strict."
        )

    user_ids = np.sort(frame["user_id"].unique())
    item_ids = np.sort(frame["place_id"].unique())
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    out = pd.DataFrame(
        {
            "user_index": frame["user_id"].map(user_map).astype(np.int64),
            "item_index": frame["place_id"].map(item_map).astype(np.int64),
            "rating": frame["rating"].astype(np.float64),
        }
    ).sort_values(["user_index", "item_index"]).reset_index(drop=True)

    return InteractionData(frame=out, user_ids=user_ids, item_ids=item_ids)
