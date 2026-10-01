"""RecomMind command-line interface.

Commands
--------
recommend --user 7 --n 5 [--method item]   top-N picks for a user
evaluate [--test-frac 0.2]                 RMSE/MAE table: user-CF vs item-CF vs popularity
demo                                       end-to-end: generate -> train -> recommend -> evaluate
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from recommind.data import generate_ratings, load_csvs, train_test_split
from recommind.evaluate import evaluate
from recommind.models import ItemBasedCF, PopularityBaseline, UserBasedCF, recommend

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(_PKG_DIR), "data")

METHODS = {
    "user": ("User-based CF", UserBasedCF),
    "item": ("Item-based CF", ItemBasedCF),
    "popularity": ("Popularity baseline", PopularityBaseline),
}


def _data_dir(explicit: str | None) -> str:
    return explicit or _DEFAULT_DATA_DIR


def _load_or_generate(data_dir: str | None, seed: int = 42):
    """Load the bundled CSVs if present, else generate fresh synthetic data."""
    ddir = _data_dir(data_dir)
    ratings_csv = os.path.join(ddir, "ratings.csv")
    items_csv = os.path.join(ddir, "items.csv")
    if os.path.exists(ratings_csv) and os.path.exists(items_csv):
        return load_csvs(ratings_csv, items_csv), False
    return generate_ratings(seed=seed), True


def _fit_all(train):
    models = {}
    for key, (name, cls) in METHODS.items():
        models[key] = cls().fit(train)
    return models


def cmd_recommend(args) -> int:
    data, generated = _load_or_generate(args.data)
    train, _ = train_test_split(data["matrix"], test_frac=0.2, seed=42)
    name, cls = METHODS[args.method]
    model = cls().fit(train)

    user = args.user
    recs = recommend(model, user, train, n=args.n)
    titles = data["titles"]
    print(f"\nTop {args.n} recommendations for user {user}  [{name}]")
    print("-" * 64)
    for rank, (item_id, score) in enumerate(recs, 1):
        print(f"  {rank}. {titles[item_id]:<38}  predicted {score:.2f}/5")
    if generated:
        print("\n(note: bundled demo CSVs not found, used fresh synthetic data)")
    return 0


def _results_table(results: dict) -> str:
    lines = [
        "",
        f"{'Method':<22} {'RMSE':>8} {'MAE':>8} {'n':>7}",
        "-" * 50,
    ]
    for key, (name, _cls) in METHODS.items():
        r = results[key]
        lines.append(f"{name:<22} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['n']:>7}")
    lines.append("")
    return "\n".join(lines)


def cmd_evaluate(args) -> int:
    data, _ = _load_or_generate(args.data)
    train, test = train_test_split(data["matrix"], test_frac=args.test_frac, seed=42)
    results = {key: evaluate(cls().fit(train), train, test) for key, (_n, cls) in METHODS.items()}
    print(f"\nHoldout evaluation  (test_frac={args.test_frac}, n={results['user']['n']} ratings)")
    print(_results_table(results))
    best = min(results, key=lambda k: results[k]["rmse"])
    print(f"Best RMSE: {METHODS[best][0]} ({results[best]['rmse']:.4f})\n")
    return 0


def cmd_demo(args) -> int:
    print("RecomMind demo: generate -> train -> recommend -> evaluate\n")

    print("[1/4] Generating synthetic MovieLens-style data (seed=42) ...")
    data = generate_ratings(n_users=200, n_items=80, seed=42)
    n_ratings = int(np.sum(~np.isnan(data["matrix"])))
    print(f"      {n_ratings} ratings | {data['n_users']} users x {data['n_items']} items")

    print("[2/4] Splitting 80/20 train/test, training 3 models ...")
    train, test = train_test_split(data["matrix"], test_frac=0.2, seed=42)
    models = _fit_all(train)
    print("      trained: " + ", ".join(METHODS[k][0] for k in METHODS))

    print("[3/4] Recommending for user 7 ...")
    recs = recommend(models["item"], 7, train, n=5)
    for rank, (item_id, score) in enumerate(recs, 1):
        print(f"      {rank}. {data['titles'][item_id]:<38}  {score:.2f}/5")

    print("[4/4] Evaluating on the holdout set ...")
    results = {key: evaluate(models[key], train, test) for key in METHODS}
    print(_results_table(results))
    print("Done. Run `recommend --user <id> --n <k>` or `evaluate` to explore.\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="recommend",
        description="RecomMind: a recommendation engine built from scratch.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("recommend", help="Top-N recommendations for a user.")
    pr.add_argument("--user", type=int, required=True, help="User id (e.g. 7)")
    pr.add_argument("--n", type=int, default=5, help="How many recommendations")
    pr.add_argument("--method", choices=list(METHODS), default="item",
                    help="Which model to use (default: item)")
    pr.add_argument("--data", default=None, help="Directory with ratings.csv/items.csv")
    pr.set_defaults(func=cmd_recommend)

    pe = sub.add_parser("evaluate", help="RMSE/MAE comparison of all methods.")
    pe.add_argument("--test-frac", type=float, default=0.2)
    pe.add_argument("--data", default=None, help="Directory with ratings.csv/items.csv")
    pe.set_defaults(func=cmd_evaluate)

    pd = sub.add_parser("demo", help="End-to-end demo: generate, train, recommend, evaluate.")
    pd.set_defaults(func=cmd_demo)

    return p


def main(argv=None) -> int:
    import sys

    if argv is None:
        argv = sys.argv[1:]
    # `recommend --user 7 --n 5` should just work: when the first arg is not
    # a subcommand, assume the `recommend` subcommand was meant.
    if argv and argv[0] not in ("recommend", "evaluate", "demo", "-h", "--help"):
        argv = ["recommend"] + argv
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
