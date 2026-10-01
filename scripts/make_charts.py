"""Generate assets/rmse_chart.png: real RMSE/MAE comparison bar chart.

Run from the repo root:  python3 scripts/make_charts.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recommind.cli import METHODS, _load_or_generate
from recommind.data import train_test_split
from recommind.evaluate import evaluate

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS, exist_ok=True)


def main():
    data, _ = _load_or_generate(None)
    train, test = train_test_split(data["matrix"], test_frac=0.2, seed=42)
    names, rmses, maes = [], [], []
    for key, (name, cls) in METHODS.items():
        r = evaluate(cls().fit(train), train, test)
        names.append(name.replace(" baseline", ""))
        rmses.append(r["rmse"])
        maes.append(r["mae"])
        print(f"{name}: RMSE={r['rmse']:.4f} MAE={r['mae']:.4f}")

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar(x - width / 2, rmses, width, label="RMSE", color="#4C78A8")
    b2 = ax.bar(x + width / 2, maes, width, label="MAE", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Error (lower is better)")
    ax.set_title("RecomMind: holdout error by method (699 test ratings)")
    ax.legend()
    ax.set_ylim(0, max(rmses) * 1.35)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = os.path.join(ASSETS, "rmse_chart.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
