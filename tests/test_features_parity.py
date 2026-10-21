"""Offline/online parity: the published top-N payload must match the ranking
the offline evaluation actually scored.
"""

from __future__ import annotations

from recommender.models import EASERecommender
from recommender.serving.topn import build_topn_payloads


def test_topn_payload_matches_model_recommend(interactions, split):
    model = EASERecommender(l2_reg=100.0).fit(split)
    payloads = build_topn_payloads(
        model, split, interactions.item_ids, interactions.user_ids, k=5, users=[0]
    )
    history = split.user_positives().get(0, set())
    expected = model.recommend(0, k=5, exclude=history)
    expected_place_ids = [str(int(interactions.item_ids[i])) for i in expected]

    served_place_ids = [r["place_id"] for r in payloads[0]["recommendations"]]
    assert served_place_ids == expected_place_ids
