"""Cosine similarity, implemented from scratch.

The cosine similarity between two vectors a and b is

    cos(a, b) = (a . b) / (||a|| * ||b||)

Zero vectors are guarded: similarity with a zero vector is defined as 0.0.
"""

import numpy as np


def cosine_sim(a, b) -> float:
    """Cosine similarity between two 1-D vectors. Returns a float in [-1, 1]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def masked_cosine(a, b, mask) -> float:
    """Cosine similarity computed only over entries where ``mask`` is True."""
    a = np.asarray(a, dtype=float)[mask]
    b = np.asarray(b, dtype=float)[mask]
    return cosine_sim(a, b)
