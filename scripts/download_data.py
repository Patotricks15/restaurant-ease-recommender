"""Fetch the Restaurant Consumer Ratings dataset, or generate a synthetic fixture.

Real data (requires ``KAGGLE_USERNAME``/``KAGGLE_KEY``):

    python scripts/download_data.py --kaggle

Synthetic fixture with the same schema, no Kaggle account needed:

    python scripts/download_data.py --synthetic
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KAGGLE_REF = "uciml/restaurant-data-with-consumer-ratings"

CUISINES = ["Mexican", "Bar", "Cafeteria", "Pizzeria", "Fast_Food", "Seafood", "Italian"]
PRICES = ["low", "medium", "high"]


def generate_synthetic(
    n_users: int = 150, n_places: int = 60, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate ratings and a restaurant catalogue with the real schema."""
    rng = np.random.default_rng(seed)

    places = pd.DataFrame(
        {
            "place_id": np.arange(1, n_places + 1),
            "cuisine": rng.choice(CUISINES, size=n_places),
            "price": rng.choice(PRICES, size=n_places, p=[0.5, 0.35, 0.15]),
        }
    )

    # Popularity-skewed interaction counts, Zipf-like, so EASE^R has a real
    # co-occurrence signal instead of uniform noise.
    popularity = rng.pareto(a=1.5, size=n_places) + 1.0
    popularity /= popularity.sum()

    rows = []
    for user_id in range(1, n_users + 1):
        n_visits = rng.integers(3, 15)
        visited = rng.choice(n_places, size=n_visits, replace=False, p=popularity)
        for place_idx in visited:
            rating = rng.choice([0, 1, 1, 2, 2], size=1)[0]
            rows.append(
                {
                    "user_id": user_id,
                    "place_id": int(places.iloc[place_idx]["place_id"]),
                    "rating": int(rating),
                }
            )

    ratings = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "place_id"])
    return ratings, places


def download_kaggle() -> None:
    """Download the real dataset via the Kaggle CLI (requires credentials)."""
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_REF, "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    rating_final = DATA_DIR / "rating_final.csv"
    geoplaces = DATA_DIR / "geoplaces2.csv"
    if not rating_final.exists() or not geoplaces.exists():
        raise FileNotFoundError("Expected rating_final.csv and geoplaces2.csv after download.")

    ratings = pd.read_csv(rating_final).rename(
        columns={"userID": "user_id", "placeID": "place_id", "rating": "rating"}
    )[["user_id", "place_id", "rating"]]
    ratings.to_csv(DATA_DIR / "ratings.csv", index=False)

    places = pd.read_csv(geoplaces).rename(columns={"placeID": "place_id"})
    places = places[["place_id", "price"]].assign(cuisine="unknown")
    places.to_csv(DATA_DIR / "places.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true")
    group.add_argument("--kaggle", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if args.synthetic:
        ratings, places = generate_synthetic()
        ratings.to_csv(DATA_DIR / "ratings.csv", index=False)
        places.to_csv(DATA_DIR / "places.csv", index=False)
        print(f"wrote {len(ratings)} ratings and {len(places)} restaurants to {DATA_DIR}")
    else:
        download_kaggle()
        print(f"downloaded and normalized the real dataset into {DATA_DIR}")


if __name__ == "__main__":
    sys.exit(main())
