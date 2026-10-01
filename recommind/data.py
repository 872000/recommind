"""Synthetic MovieLens-style ratings dataset generator.

Ratings are produced from a latent-factor model (users and items each get
a hidden feature vector, plus genre preferences), so the data has the
low-rank-ish structure that makes collaborative filtering work. Everything
is seeded, so the same seed always produces the same dataset.
"""

from __future__ import annotations

import numpy as np

GENRES = [
    "Action",
    "Comedy",
    "Drama",
    "Sci-Fi",
    "Horror",
    "Romance",
    "Thriller",
    "Documentary",
]

_ADJECTIVES = [
    "Midnight", "Silent", "Electric", "Crimson", "Golden", "Frozen",
    "Velvet", "Neon", "Hollow", "Burning", "Whispering", "Rising",
]
_NOUNS = [
    "Horizon", "Echo", "Kingdom", "Voyage", "Cipher", "Garden",
    "Signal", "Empire", "Harbor", "Mirage", "Anthem", "Frontier",
]


def make_titles(n_items: int, genres: np.ndarray, seed: int = 42) -> list[str]:
    """Deterministic, human-friendly item titles built from genre + word lists."""
    rng = np.random.default_rng(seed + 999)
    titles = []
    for i in range(n_items):
        adj = _ADJECTIVES[rng.integers(0, len(_ADJECTIVES))]
        noun = _NOUNS[rng.integers(0, len(_NOUNS))]
        titles.append(f"The {adj} {noun} ({GENRES[int(genres[i])]})")
    return titles


def generate_ratings(
    n_users: int = 200,
    n_items: int = 80,
    n_factors: int = 6,
    density: float = 0.22,
    noise: float = 0.3,
    latent_scale: float = 1.3,
    genre_scale: float = 0.5,
    seed: int = 42,
) -> dict:
    """Generate a synthetic ratings matrix.

    Returns a dict with:
      - ``matrix``: (n_users, n_items) float array; NaN = not rated.
      - ``genres``: (n_items,) int array of genre indices into GENRES.
      - ``titles``: list of item titles.
      - ``n_users``, ``n_items``.

    The rating model is::

        r_ui = 3.0 + latent_scale * (u_f . i_f) / sqrt(k)
                 + genre_scale * genre_pref + noise

    clipped to [1, 5] and rounded to the nearest 0.5 (MovieLens style).
    Each user rates at least 5 items and each item is rated at least
    3 times, so neighbourhood methods always have something to chew on.
    """
    rng = np.random.default_rng(seed)

    user_factors = rng.normal(0.0, 1.0, (n_users, n_factors))
    item_factors = rng.normal(0.0, 1.0, (n_items, n_factors))
    genres = rng.integers(0, len(GENRES), n_items)
    user_genre_pref = rng.normal(0.0, 0.8, (n_users, len(GENRES)))

    latent = (user_factors @ item_factors.T) / np.sqrt(n_factors)
    genre_effect = user_genre_pref[np.arange(n_users)[:, None], genres[None, :]]
    true_scores = 3.0 + latent_scale * latent + genre_scale * genre_effect
    true_scores += rng.normal(0.0, noise, (n_users, n_items))
    true_scores = np.clip(np.round(true_scores * 2.0) / 2.0, 1.0, 5.0)

    # Observation mask: each user rates ~density of the catalog, min 5 items.
    observed = rng.random((n_users, n_items)) < density
    for u in range(n_users):
        while observed[u].sum() < 5:
            observed[u, rng.integers(0, n_items)] = True
    for i in range(n_items):
        while observed[:, i].sum() < 3:
            observed[rng.integers(0, n_users), i] = True

    matrix = np.where(observed, true_scores, np.nan)
    return {
        "matrix": matrix,
        "genres": genres,
        "titles": make_titles(n_items, genres, seed),
        "n_users": n_users,
        "n_items": n_items,
    }


def train_test_split(matrix: np.ndarray, test_frac: float = 0.2, seed: int = 42):
    """Split observed ratings into train/test, both NaN-masked matrices.

    A fraction ``test_frac`` of the observed (user, item) pairs is held out.
    The split is seeded and reproducible.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(~np.isnan(matrix))
    pairs = np.stack([rows, cols], axis=1)
    rng.shuffle(pairs)
    n_test = max(1, int(len(pairs) * test_frac))
    test_pairs = pairs[:n_test]

    train = matrix.copy()
    test = np.full_like(matrix, np.nan)
    for u, i in test_pairs:
        test[u, i] = matrix[u, i]
        train[u, i] = np.nan
    return train, test


def ratings_to_triples(matrix: np.ndarray):
    """Yield (user_id, item_id, rating) for every observed rating."""
    rows, cols = np.nonzero(~np.isnan(matrix))
    for u, i in zip(rows.tolist(), cols.tolist()):
        yield int(u), int(i), float(matrix[u, i])


def save_csvs(data: dict, ratings_path: str, items_path: str) -> None:
    """Write the demo bundle: ratings.csv + items.csv (plain csv module)."""
    import csv

    with open(ratings_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "rating"])
        for u, i, r in ratings_to_triples(data["matrix"]):
            w.writerow([u, i, f"{r:.1f}"])

    with open(items_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "title", "genre"])
        for i, title in enumerate(data["titles"]):
            w.writerow([i, title, GENRES[int(data["genres"][i])]])


def load_csvs(ratings_path: str, items_path: str) -> dict:
    """Load a ratings.csv + items.csv bundle back into the dict format."""
    import csv

    with open(items_path, newline="") as f:
        item_rows = list(csv.DictReader(f))
    n_items = len(item_rows)
    titles = [row["title"] for row in item_rows]
    genres = np.array([GENRES.index(row["genre"]) for row in item_rows])

    with open(ratings_path, newline="") as f:
        rating_rows = list(csv.DictReader(f))
    n_users = max(int(row["user_id"]) for row in rating_rows) + 1
    matrix = np.full((n_users, n_items), np.nan)
    for row in rating_rows:
        matrix[int(row["user_id"]), int(row["item_id"])] = float(row["rating"])

    return {
        "matrix": matrix,
        "genres": genres,
        "titles": titles,
        "n_users": n_users,
        "n_items": n_items,
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import os

    p = argparse.ArgumentParser(description="Generate the bundled demo dataset.")
    p.add_argument("--out", default="data", help="Directory for ratings.csv/items.csv")
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--items", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = generate_ratings(n_users=args.users, n_items=args.items, seed=args.seed)
    save_csvs(
        data,
        os.path.join(args.out, "ratings.csv"),
        os.path.join(args.out, "items.csv"),
    )
    n_ratings = int(np.sum(~np.isnan(data["matrix"])))
    print(f"Wrote {n_ratings} ratings for {args.users} users x {args.items} items -> {args.out}/")
