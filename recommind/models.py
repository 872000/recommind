"""Collaborative filtering models, from scratch.

- ``UserBasedCF``: predicts a rating from the k most similar users
  (mean-centered cosine similarity over co-rated items).
- ``ItemBasedCF``: predicts a rating from items the user already rated,
  weighted by item-item cosine similarity (over co-rating users).
- ``PopularityBaseline``: predicts each item's mean rating; used as the
  baseline and as the cold-start fallback for unknown users/items.

All models share one interface: ``fit(train_matrix)`` then
``predict(user_id, item_id) -> float``. Ratings are clipped to [1, 5].
"""

from __future__ import annotations

import warnings

import numpy as np

from recommind.similarity import cosine_sim

_MIN_RATING, _MAX_RATING = 1.0, 5.0


def _pairwise_cosine(centered: np.ndarray, axis: int) -> np.ndarray:
    """Cosine similarity between rows (axis=0) or columns (axis=1) of an
    already mean-centered matrix, computed over co-observed entries only.

    Centering removes per-user rating bias, so similarity reflects
    correlated *taste* rather than correlated *generosity* (this is the
    "adjusted cosine" used by textbook item-based CF).
    """
    mat = centered if axis == 0 else centered.T
    n = mat.shape[0]
    filled = np.where(np.isnan(mat), 0.0, mat)
    rated = ~np.isnan(mat)

    sim = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            co = rated[a] & rated[b]
            if not np.any(co):
                continue
            s = cosine_sim(filled[a][co], filled[b][co])
            sim[a, b] = sim[b, a] = s
    return sim


def _center_by_user_mean(train: np.ndarray, user_means: np.ndarray) -> np.ndarray:
    """Subtract each user's mean rating; NaN (unrated) stays NaN."""
    return train - user_means[:, None]


def _clip(x: float) -> float:
    return float(min(_MAX_RATING, max(_MIN_RATING, x)))


class UserBasedCF:
    """User-based collaborative filtering with k nearest neighbours."""

    def __init__(self, k: int = 25):
        self.k = k

    def fit(self, train: np.ndarray):
        self.train = np.asarray(train, dtype=float)
        self.n_users, self.n_items = self.train.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            self.global_mean = float(np.nanmean(self.train))
            self.user_means = np.nanmean(self.train, axis=1)
        self.user_means = np.where(np.isnan(self.user_means), self.global_mean, self.user_means)
        self.sim = _pairwise_cosine(_center_by_user_mean(self.train, self.user_means), axis=0)
        return self

    def _cold_start(self, u: int, i: int) -> float:
        """Popularity fallback for unknown users, users with no ratings,
        items with no ratings, or empty neighbourhoods."""
        if 0 <= i < self.n_items:
            item_mean = np.nanmean(self.train[:, i])
            if not np.isnan(item_mean):
                return _clip(item_mean)
        return _clip(self.global_mean)

    def predict(self, user_id: int, item_id: int) -> float:
        if not (0 <= user_id < self.n_users) or np.isnan(self.user_means[user_id]):
            return self._cold_start(user_id, item_id)

        rated_by = np.nonzero(~np.isnan(self.train[:, item_id]))[0]
        rated_by = rated_by[rated_by != user_id]
        if len(rated_by) == 0:
            return self._cold_start(user_id, item_id)

        sims = self.sim[user_id, rated_by]
        order = np.argsort(-np.abs(sims))[: self.k]
        neigh, sims = rated_by[order], sims[order]

        denom = np.sum(np.abs(sims))
        if denom == 0.0:
            return self._cold_start(user_id, item_id)

        dev = self.train[neigh, item_id] - self.user_means[neigh]
        pred = self.user_means[user_id] + float(np.dot(sims, dev) / denom)
        return _clip(pred)


class ItemBasedCF:
    """Item-based collaborative filtering with adjusted cosine similarity.

    pred(u, i) = mean_u + sum_j sim(i, j) * (r_uj - mean_u) / sum_j |sim(i, j)|
    over the k items j most similar to i that user u has rated, where
    sim(i, j) is the cosine of user-mean-centered ratings over users who
    rated both items.
    """

    def __init__(self, k: int = 25):
        self.k = k

    def fit(self, train: np.ndarray):
        self.train = np.asarray(train, dtype=float)
        self.n_users, self.n_items = self.train.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            self.global_mean = float(np.nanmean(self.train))
            self.user_means = np.nanmean(self.train, axis=1)
        self.user_means = np.where(np.isnan(self.user_means), self.global_mean, self.user_means)
        self.sim = _pairwise_cosine(_center_by_user_mean(self.train, self.user_means), axis=1)
        return self

    def _cold_start(self, u: int, i: int) -> float:
        if 0 <= i < self.n_items:
            item_mean = np.nanmean(self.train[:, i])
            if not np.isnan(item_mean):
                return _clip(item_mean)
        return _clip(self.global_mean)

    def predict(self, user_id: int, item_id: int) -> float:
        if not (0 <= user_id < self.n_users) or not (0 <= item_id < self.n_items):
            return self._cold_start(user_id, item_id)

        rated = np.nonzero(~np.isnan(self.train[user_id]))[0]
        rated = rated[rated != item_id]
        if len(rated) == 0:
            return self._cold_start(user_id, item_id)

        sims = self.sim[item_id, rated]
        order = np.argsort(-np.abs(sims))[: self.k]
        items, sims = rated[order], sims[order]

        denom = np.sum(np.abs(sims))
        if denom == 0.0:
            return self._cold_start(user_id, item_id)

        dev = self.train[user_id, items] - self.user_means[user_id]
        pred = self.user_means[user_id] + float(np.dot(sims, dev) / denom)
        return _clip(pred)


class PopularityBaseline:
    """The 'just show what's popular' baseline.

    Predicts each item's mean rating (global mean for unknown items).
    Doubles as the cold-start fallback used by the CF models.
    """

    def fit(self, train: np.ndarray):
        self.train = np.asarray(train, dtype=float)
        self.n_users, self.n_items = self.train.shape
        self.global_mean = float(np.nanmean(self.train))
        self.item_means = np.nanmean(self.train, axis=0)
        self.item_means = np.where(np.isnan(self.item_means), self.global_mean, self.item_means)
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        if 0 <= item_id < self.n_items:
            return _clip(float(self.item_means[item_id]))
        return _clip(self.global_mean)


def recommend(model, user_id: int, train: np.ndarray, n: int = 5):
    """Top-N item ids for ``user_id``, excluding items already rated in ``train``.

    Returns a list of ``(item_id, predicted_rating)`` sorted best-first.
    Unknown users fall back to the model's popularity predictions.
    """
    train = np.asarray(train, dtype=float)
    n_items = train.shape[1]
    seen = set()
    if 0 <= user_id < train.shape[0]:
        seen = set(np.nonzero(~np.isnan(train[user_id]))[0].tolist())

    scored = []
    for i in range(n_items):
        if i in seen:
            continue
        scored.append((i, model.predict(user_id, i)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:n]
