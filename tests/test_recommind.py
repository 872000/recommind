"""RecomMind test suite. Run with: python -m pytest  (from the repo root)."""

import math

import numpy as np
import pytest

from recommind.data import generate_ratings, train_test_split
from recommind.evaluate import evaluate, mae, rmse
from recommind.models import ItemBasedCF, PopularityBaseline, UserBasedCF, recommend
from recommind.similarity import cosine_sim


# ---------------------------------------------------------------- cosine
def test_cosine_sim_orthogonal_vectors():
    assert cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_sim_parallel_vectors():
    assert cosine_sim([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_cosine_sim_opposite_vectors():
    assert cosine_sim([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_sim_known_value():
    # cos([1,1],[1,0]) = 1 / sqrt(2)
    assert cosine_sim([1, 1], [1, 0]) == pytest.approx(1 / math.sqrt(2))


def test_cosine_sim_zero_vector_guarded():
    assert cosine_sim([0, 0, 0], [1, 2, 3]) == 0.0


# ---------------------------------------------------------------- data
def test_generate_ratings_deterministic_with_seed():
    a = generate_ratings(seed=123)
    b = generate_ratings(seed=123)
    np.testing.assert_array_equal(
        np.nan_to_num(a["matrix"]), np.nan_to_num(b["matrix"])
    )
    assert a["titles"] == b["titles"]


def test_generate_ratings_ratings_in_range():
    data = generate_ratings(n_users=50, n_items=20, seed=7)
    m = data["matrix"]
    assert np.nanmin(m) >= 1.0 and np.nanmax(m) <= 5.0


def test_train_test_split_disjoint_and_reproducible():
    data = generate_ratings(n_users=50, n_items=20, seed=7)
    t1, e1 = train_test_split(data["matrix"], test_frac=0.2, seed=1)
    t2, e2 = train_test_split(data["matrix"], test_frac=0.2, seed=1)
    # Same seed -> identical splits
    np.testing.assert_array_equal(np.nan_to_num(t1), np.nan_to_num(t2))
    np.testing.assert_array_equal(np.nan_to_num(e1), np.nan_to_num(e2))
    # Train and test never share a rating
    assert not np.any(~np.isnan(t1) & ~np.isnan(e1))
    # Union covers every original rating
    orig = ~np.isnan(data["matrix"])
    assert np.all((~np.isnan(t1) | ~np.isnan(e1)) == orig)


# ---------------------------------------------------------------- metrics
def test_rmse_perfect_predictions_is_zero():
    y = [1.0, 2.5, 4.0, 5.0]
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known_value():
    # errors 1 and 2 -> sqrt((1+4)/2) = sqrt(2.5)
    assert rmse([3.0, 4.0], [4.0, 6.0]) == pytest.approx(math.sqrt(2.5))


def test_mae_known_value():
    assert mae([3.0, 4.0], [4.0, 6.0]) == pytest.approx(1.5)


# ---------------------------------------------------------------- models
@pytest.fixture()
def small_data():
    # 6 users x 8 items, fixed seed -> stable neighbourhoods
    data = generate_ratings(n_users=30, n_items=12, density=0.4, seed=99)
    train, test = train_test_split(data["matrix"], test_frac=0.2, seed=5)
    return data, train, test


def test_models_predict_within_rating_range(small_data):
    _, train, _ = small_data
    for cls in (UserBasedCF, ItemBasedCF, PopularityBaseline):
        model = cls().fit(train)
        for u in range(train.shape[0]):
            for i in range(train.shape[1]):
                p = model.predict(u, i)
                assert 1.0 <= p <= 5.0, f"{cls.__name__} predicted {p}"


def test_cold_start_unknown_user_falls_back_to_popularity(small_data):
    _, train, _ = small_data
    unknown = train.shape[0] + 100
    for cls in (UserBasedCF, ItemBasedCF):
        model = cls().fit(train)
        base = PopularityBaseline().fit(train)
        assert model.predict(unknown, 0) == pytest.approx(base.predict(unknown, 0))


def test_recommend_returns_n_items_excluding_seen(small_data):
    _, train, _ = small_data
    model = ItemBasedCF().fit(train)
    user = 3
    n = 5
    recs = recommend(model, user, train, n=n)
    assert len(recs) == n
    seen = set(np.nonzero(~np.isnan(train[user]))[0].tolist())
    for item_id, score in recs:
        assert item_id not in seen, "recommended an already-seen item"
        assert 1.0 <= score <= 5.0
    # sorted best-first
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_rejects_negative_n(small_data):
    _, train, _ = small_data
    model = ItemBasedCF().fit(train)
    with pytest.raises(ValueError, match="non-negative.*-1"):
        recommend(model, 3, train, n=-1)


def test_recommend_excludes_seen_for_user_cf(small_data):
    _, train, _ = small_data
    model = UserBasedCF().fit(train)
    recs = recommend(model, 0, train, n=4)
    seen = set(np.nonzero(~np.isnan(train[0]))[0].tolist())
    assert len(recs) == 4
    assert all(i not in seen for i, _ in recs)


def test_item_cf_finds_similar_taste():
    # Users 0 and 1 rate items 0/1/2 above their personal means and item 3/4
    # below it; users 2 and 3 show the opposite pattern. Item-based CF with
    # adjusted cosine should therefore rank the unseen item 2 (taste-aligned
    # with items 0,1) far above item 3 for user 0.
    m = np.full((4, 6), np.nan)
    m[0, 0] = 5.0
    m[0, 1] = 4.0
    m[0, 4] = 2.0
    m[1, 0] = 5.0
    m[1, 1] = 5.0
    m[1, 2] = 4.0
    m[1, 3] = 1.0
    m[1, 4] = 2.0
    m[2, 3] = 5.0
    m[2, 4] = 4.0
    m[2, 5] = 5.0
    m[3, 0] = 1.0
    m[3, 3] = 4.0
    m[3, 5] = 5.0
    model = ItemBasedCF(k=5).fit(m)
    assert model.predict(0, 2) > model.predict(0, 3) + 1.0


def test_evaluate_returns_sane_metrics(small_data):
    _, train, test = small_data
    result = evaluate(PopularityBaseline().fit(train), train, test)
    assert result["n"] > 0
    assert 0.0 < result["rmse"] < 3.0
    assert 0.0 < result["mae"] <= result["rmse"]
    assert result["coverage"] == 1.0
