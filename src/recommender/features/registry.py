"""Registry mapping declared feature names to their transform implementations.

``assert_registry_matches_spec`` is the two-way check referenced by
``conf/features.yaml``: every declared feature must have a registered
transform, and every registered transform must be declared.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from recommender.features.spec import FeatureSpec
from recommender.features.transforms import (
    item_cuisine_tokens,
    item_popularity_rank,
    item_price_tier,
    user_activity_count,
    user_mean_rating,
)

TRANSFORMS: dict[str, Callable[..., pd.Series]] = {
    "user_activity_count": user_activity_count,
    "user_mean_rating": user_mean_rating,
    "item_popularity_rank": item_popularity_rank,
    "item_price_tier": item_price_tier,
    "item_cuisine_tokens": item_cuisine_tokens,
}


def assert_registry_matches_spec(spec: FeatureSpec) -> None:
    """Fail loudly if the spec and the registry disagree in either direction.

    Args:
        spec: The loaded declarative specification.

    Raises:
        ValueError: If a declared transform is unimplemented, or an
            implemented transform is undeclared.
    """
    declared = {f.transform for f in spec.features}
    implemented = set(TRANSFORMS)

    undeclared = implemented - declared
    unimplemented = declared - implemented
    if undeclared or unimplemented:
        raise ValueError(
            "Feature registry and conf/features.yaml disagree: "
            f"implemented-but-undeclared={sorted(undeclared)}, "
            f"declared-but-unimplemented={sorted(unimplemented)}."
        )
