"""Feature-MLP bar charts from results/<run>/history.npz: remove-one ablation and n-step sweep."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def final(run, k=300):
    return float(np.mean(np.load(ROOT / "results" / run / "history.npz")["scores"][-k:]))


def bars(ax, items, title, highlight=0):
    labels, vals = [k for k, _ in items], [v for _, v in items]
    colors = ["tab:orange"] * len(items)
    colors[highlight] = "tab:red"
    rects = ax.bar(labels, vals, color=colors)
    for rect, v in zip(rects, vals, strict=True):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Food eaten (avg of last 300 episodes)")
    ax.set_title(title)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars(axes[0], [("full", final("feat_full")), ("- n-step", final("feat_full_no_nstep")),
                   ("- PBRS", final("feat_full_no_pbrs")), ("- Double", final("feat_full_no_double"))],
         "Feature MLP: remove one component from full")
    bars(axes[1], [("1-step", final("feat_full_no_nstep")), ("3-step (full)", final("feat_full")),
                   ("5-step", final("feat_full_n5")), ("7-step", final("feat_full_n7")), ("10-step", final("feat_full_n10"))],
         "Feature MLP: n-step return length", highlight=1)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/feat_ablation.png", dpi=150)


if __name__ == "__main__":
    main()
