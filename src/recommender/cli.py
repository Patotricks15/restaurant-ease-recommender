"""Command-line entry points for training, evaluation, and recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommender.data import build_implicit_feedback, load_ratings, user_holdout_split
from recommender.evaluation import evaluate
from recommender.models import EASERecommender, PopularityRecommender

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_L2_REG = 300.0


def _load():
    ratings = load_ratings(ROOT / "data" / "ratings.csv")
    data = build_implicit_feedback(ratings)
    return data, user_holdout_split(data)


def train_command(_: argparse.Namespace) -> None:
    _, split = _load()
    model = EASERecommender(l2_reg=DEFAULT_L2_REG).fit(split)
    print(json.dumps({"l2_reg": model.l2_reg, "n_items": model.n_items, "n_users": model.n_users}))


def evaluate_command(args: argparse.Namespace) -> None:
    _, split = _load()
    models = {
        "EASE": EASERecommender(l2_reg=DEFAULT_L2_REG).fit(split),
        "Popularity": PopularityRecommender().fit(split),
    }
    rows = [
        evaluate(model, split, k=args.k, partition=args.partition, model_name=name).as_row()
        for name, model in models.items()
    ]
    print(json.dumps(rows, indent=2))


def recommend_command(args: argparse.Namespace) -> None:
    data, split = _load()
    matches = (data.user_ids == args.user_id).nonzero()[0]
    if matches.size == 0:
        raise SystemExit(f"Unknown user id {args.user_id}.")
    user = int(matches[0])
    model = EASERecommender(l2_reg=DEFAULT_L2_REG).fit(split)
    history = split.user_positives().get(user, set())
    ranked = model.recommend(user, k=args.k, exclude=history)

    payload = [
        {"rank": rank, "place_id": int(data.item_ids[item])}
        for rank, item in enumerate(ranked, start=1)
    ]
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.set_defaults(func=train_command)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--partition", choices=["validation", "test"], default="test")
    evaluate_parser.add_argument("--k", type=int, default=10)
    evaluate_parser.set_defaults(func=evaluate_command)

    recommend = commands.add_parser("recommend")
    recommend.add_argument("--user-id", type=int, required=True)
    recommend.add_argument("--k", type=int, default=10)
    recommend.set_defaults(func=recommend_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
