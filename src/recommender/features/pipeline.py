"""Build the user/item feature tables declared in ``conf/features.yaml``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.splitting import Split
from recommender.features.registry import TRANSFORMS, assert_registry_matches_spec
from recommender.features.spec import FeatureSpec


@dataclass(frozen=True)
class FeatureTables:
    """Materialized feature tables, indexed by internal item/user index."""

    user_features: pd.DataFrame
    item_features: pd.DataFrame


def build_feature_tables(
    split: Split, catalogue: pd.DataFrame, item_ids: np.ndarray, spec: FeatureSpec
) -> FeatureTables:
    """Materialize every declared feature, strictly from training-time data.

    Args:
        split: Leakage-safe partition; only ``split.train`` feeds
            ``fit_on: train_only`` features.
        catalogue: Restaurant metadata frame (``place_id``, ``price``,
            ``cuisine``), indexed externally by ``place_id``.
        item_ids: Original ``place_id``s, positionally aligned to internal
            item index; used to reindex catalogue-sourced columns into the
            same index space as the training-derived columns.
        spec: The loaded declarative feature specification.

    Returns:
        User and item feature tables, both indexed by internal index.

    Raises:
        ValueError: If the registry and the spec disagree.
    """
    assert_registry_matches_spec(spec)

    user_columns, item_columns = {}, {}
    for feature in spec.features:
        transform = TRANSFORMS[feature.transform]
        if feature.family == "user":
            user_columns[feature.name] = transform(split)
        elif feature.fit_on == "train_only":
            item_columns[feature.name] = transform(split)
        else:
            # Catalogue transforms return a place_id-indexed series; realign
            # to the internal item index so every item column shares one
            # index space.
            raw = transform(catalogue)
            item_columns[feature.name] = raw.reindex(item_ids).reset_index(drop=True)

    user_features = pd.DataFrame(user_columns)
    item_features = pd.DataFrame(item_columns)
    return FeatureTables(user_features=user_features, item_features=item_features)
