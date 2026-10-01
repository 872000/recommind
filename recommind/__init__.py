"""RecomMind: recommendation engine from scratch.

No sklearn, no black boxes. Pure Python + numpy implementations of
user-based and item-based collaborative filtering with cosine similarity,
a popularity baseline for cold start, and from-scratch RMSE/MAE evaluation.
"""

from recommind.data import generate_ratings, train_test_split, GENRES
from recommind.similarity import cosine_sim
from recommind.models import UserBasedCF, ItemBasedCF, PopularityBaseline, recommend
from recommind.evaluate import rmse, mae, evaluate

__all__ = [
    "generate_ratings",
    "train_test_split",
    "GENRES",
    "cosine_sim",
    "UserBasedCF",
    "ItemBasedCF",
    "PopularityBaseline",
    "recommend",
    "rmse",
    "mae",
    "evaluate",
]

__version__ = "1.0.0"
