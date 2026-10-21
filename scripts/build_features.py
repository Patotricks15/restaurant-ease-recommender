"""Build and persist the declarative feature tables.

    python scripts/build_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import build_implicit_feedback, load_ratings, user_holdout_split  # noqa: E402
from recommender.features import build_feature_tables, load_feature_spec  # noqa: E402


def main() -> None:
    ratings = load_ratings(ROOT / "data" / "ratings.csv")
    places = __import__("pandas").read_csv(ROOT / "data" / "places.csv")
    data = build_implicit_feedback(ratings)
    split = user_holdout_split(data)
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")

    tables = build_feature_tables(split, places, data.item_ids, spec)

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    tables.user_features.to_parquet(out_dir / "features_users.parquet")
    tables.item_features.to_parquet(out_dir / "features_items.parquet")
    print(
        f"wrote {tables.user_features.shape} user features and "
        f"{tables.item_features.shape} item features to {out_dir}"
    )


if __name__ == "__main__":
    main()
