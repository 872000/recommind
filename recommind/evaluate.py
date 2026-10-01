"""Evaluation metrics from scratch: RMSE and MAE on a holdout split."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error: sqrt(mean((true - pred)^2))."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error: mean(|true - pred|)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate(model, train: np.ndarray, test: np.ndarray) -> dict:
    """Score a fitted model on every observed rating in ``test``.

    Returns ``{"rmse", "mae", "n", "coverage"}`` where coverage is the
    fraction of test ratings the model could predict (here always 1.0,
    since every model has a cold-start fallback).
    """
    rows, cols = np.nonzero(~np.isnan(test))
    y_true, y_pred = [], []
    for u, i in zip(rows.tolist(), cols.tolist()):
        y_true.append(float(test[u, i]))
        y_pred.append(float(model.predict(int(u), int(i))))
    n = len(y_true)
    return {
        "rmse": rmse(y_true, y_pred) if n else float("nan"),
        "mae": mae(y_true, y_pred) if n else float("nan"),
        "n": n,
        "coverage": 1.0,
    }
