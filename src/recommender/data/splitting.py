"""Dataset partitioning.

The Restaurant Consumer Ratings dataset carries no event timestamp, so a
temporal split is not possible. Instead this project uses a **user-aware
random holdout**: for each user with enough history, a fraction of their
interactions is held out for validation/test, and the rest stays in
training. Users with too little history to split safely contribute entirely
to training and are not evaluated (reported as the cold-user rate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.loader import InteractionData


@dataclass(frozen=True)
class Split:
    """Train/validation/test partitions over contiguous indices."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    protocol: str = "user_holdout"

    def train_matrix_entries(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(users, items)`` arrays for the training partition."""
        return (
            self.train["user_index"].to_numpy(dtype=np.int64),
            self.train["item_index"].to_numpy(dtype=np.int64),
        )

    def user_positives(self) -> dict[int, set[int]]:
        """Map each user to the set of restaurants seen in training."""
        return {
            int(user): set(map(int, items))
            for user, items in self.train.groupby("user_index")["item_index"]
        }

    def __post_init__(self) -> None:
        if self.train.empty:
            raise ValueError("The training partition is empty.")


def user_holdout_split(
    data: InteractionData,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    min_train_interactions: int = 2,
    random_state: int = 42,
) -> Split:
    """Hold out a per-user fraction of interactions for validation and test.

    Args:
        data: Interactions with contiguous indices.
        validation_fraction: Fraction of each eligible user's interactions
            moved to validation.
        test_fraction: Fraction of each eligible user's interactions moved
            to test.
        min_train_interactions: Users must keep at least this many
            interactions in training; otherwise all of their interactions
            stay in training and they are excluded from evaluation.
        random_state: Seed controlling which interactions are held out.

    Returns:
        A :class:`Split` with ``protocol="user_holdout"``.

    Raises:
        ValueError: If the fractions are not in ``(0, 1)`` and summing to
            less than 1.
    """
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("validation_fraction and test_fraction must be in (0, 1).")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1.")

    rng = np.random.default_rng(random_state)
    train_rows, val_rows, test_rows = [], [], []

    for _, group in data.frame.groupby("user_index"):
        n = len(group)
        n_held = int(np.floor(n * (validation_fraction + test_fraction)))
        if n - n_held < min_train_interactions or n_held < 2:
            train_rows.append(group)
            continue

        shuffled = group.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1))
        n_test = max(1, int(np.floor(n * test_fraction)))
        n_val = max(1, n_held - n_test)

        test_rows.append(shuffled.iloc[:n_test])
        val_rows.append(shuffled.iloc[n_test : n_test + n_val])
        train_rows.append(shuffled.iloc[n_test + n_val :])

    train = pd.concat(train_rows).reset_index(drop=True) if train_rows else data.frame.iloc[0:0]
    validation = pd.concat(val_rows).reset_index(drop=True) if val_rows else data.frame.iloc[0:0]
    test = pd.concat(test_rows).reset_index(drop=True) if test_rows else data.frame.iloc[0:0]

    return Split(
        train=train,
        validation=validation,
        test=test,
        n_users=data.n_users,
        n_items=data.n_items,
        protocol="user_holdout",
    )


def cold_user_rate(data: InteractionData, split: Split) -> float:
    """Fraction of users excluded from evaluation for having too little history."""
    all_users = set(data.frame["user_index"].unique())
    evaluated_users = set(split.validation["user_index"].unique()) | set(
        split.test["user_index"].unique()
    )
    if not all_users:
        return 0.0
    return 1.0 - len(evaluated_users) / len(all_users)
