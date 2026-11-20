from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#1f4e79", "#c0392b", "#27865c", "#b8860b", "#6a4c93", "#2c3e50"]


def _style(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.25, linewidth=0.6)


def save(figure: plt.Figure, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        target = path.with_suffix(f".{suffix}")
        figure.savefig(target, dpi=200, bbox_inches="tight")
        written.append(target)
    plt.close(figure)
    return written


def plateau_figure(frame, path: Path, entropy_column: str = "ks_entropy") -> list[Path]:
    """Information rate against the Kolmogorov-Sinai entropy, one series per precision."""
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    precisions = sorted(frame["precision"].unique(), reverse=True)
    for index, precision in enumerate(precisions):
        subset = frame[frame["precision"] == precision].sort_values(entropy_column)
        axis.plot(
            subset[entropy_column],
            subset["information_rate"],
            marker="o",
            markersize=4.5,
            linewidth=1.2,
            color=PALETTE[index % len(PALETTE)],
            label=f"precision {precision:g}",
        )
    limit = float(np.nanmax(frame[entropy_column])) * 1.05 + 1e-6
    axis.plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1.0, label="identity, predicted law")
    axis.set_xlabel("Kolmogorov-Sinai entropy, nats per unit time")
    axis.set_ylabel("measured information rate, nats per unit time")
    axis.set_yscale("log")
    axis.legend(frameon=False, fontsize=8)
    _style(axis)
    return save(figure, path)


def precision_sweep_figure(frame, path: Path, value_column: str = "information_rate") -> list[Path]:
    """Rate against precision, one line per system. Flat means threshold independent."""
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for index, system in enumerate(sorted(frame["system"].unique())):
        subset = frame[frame["system"] == system].sort_values("precision", ascending=False)
        axis.plot(
            1.0 / subset["precision"],
            subset[value_column],
            marker="o",
            markersize=4.0,
            linewidth=1.1,
            color=PALETTE[index % len(PALETTE)],
            label=system,
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("inverse precision")
    axis.set_ylabel(value_column.replace("_", " "))
    axis.legend(frameon=False, fontsize=7, ncol=2)
    _style(axis)
    return save(figure, path)
