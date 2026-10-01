"""Generate assets/cli_demo.png: a terminal-style screenshot rendered from
REAL `recommend --user 7 --n 5` output (no mockups).

Run from the repo root:  python3 scripts/make_cli_screenshot.py
"""

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recommind.cli import main as cli_main

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS, exist_ok=True)

BG = "#1b1e28"
FG = "#d7dce2"
PROMPT_FG = "#7ee787"
HEADER_FG = "#79c0ff"


def main():
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_main(["recommend", "--user", "7", "--n", "5"])
    body = buf.getvalue().strip("\n").split("\n")

    lines = [("$ recommend --user 7 --n 5", PROMPT_FG)]
    for i, ln in enumerate(body):
        color = HEADER_FG if i <= 1 else FG
        lines.append((ln if ln else " ", color))

    fig_w, fig_h = 10, 3.2 + 0.42 * len(lines)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    y = 0.96
    dy = 0.88 / max(len(lines), 1)
    for text, color in lines:
        ax.text(0.03, y, text, transform=ax.transAxes, color=color,
                fontsize=12.5, family="monospace", va="top", ha="left")
        y -= dy

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # window chrome dots
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        ax.add_patch(plt.Circle((0.035 + i * 0.035, 0.985), 0.012, color=c,
                                transform=ax.transAxes, clip_on=False))
    out = os.path.join(ASSETS, "cli_demo.png")
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight", pad_inches=0.4)
    print("wrote", out)
    print("captured", len(body), "lines of real CLI output")


if __name__ == "__main__":
    main()
