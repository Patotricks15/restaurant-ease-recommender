"""Leakage-prevention tests for the feature engineering layer."""

from __future__ import annotations

from pathlib import Path

from recommender.features import build_feature_tables, load_feature_spec
from recommender.features.registry import assert_registry_matches_spec

ROOT = Path(__file__).resolve().parents[1]


def test_user_features_use_only_training_interactions(interactions, split, catalogue):
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    tables = build_feature_tables(split, catalogue, interactions.item_ids, spec)
    # A user absent from training has no row at all: the feature cannot have
    # been computed from data the model never saw.
    train_users = set(split.train["user_index"].unique())
    assert set(tables.user_features.index).issubset(train_users)


def test_item_price_tier_is_catalogue_only_not_label_derived(interactions, split, catalogue):
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    item_ids = interactions.item_ids
    tables = build_feature_tables(split, catalogue, item_ids, spec)
    # Catalogue-sourced columns must match the catalogue verbatim: no
    # interaction/label signal can have leaked in.
    expected = catalogue.set_index("place_id")["price"].reindex(item_ids).reset_index(drop=True)
    actual = tables.item_features["item_price_tier"].astype(object).reset_index(drop=True)
    assert (actual.values == expected.values).all()


def test_registry_matches_spec_declaration():
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    assert_registry_matches_spec(spec)  # must not raise
