from __future__ import annotations

import numpy as np


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares fit with the statistics a reviewer will ask for."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r_squared": float("nan"), "n": int(x.size)}
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = y - y.mean()
    r_squared = 1.0 - float(residual @ residual) / float(total @ total) if total @ total > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r_squared": r_squared, "n": int(x.size)}


def coding_gap(information_rate: np.ndarray, ks_entropy: np.ndarray) -> np.ndarray:
    """Excess information rate over the theoretical floor, in nats per unit time."""
    return np.asarray(information_rate, dtype=float) - np.asarray(ks_entropy, dtype=float)


def coding_ratio(information_rate: np.ndarray, ks_entropy: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    return np.asarray(information_rate, dtype=float) / np.maximum(np.asarray(ks_entropy, dtype=float), floor)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    rank_x = np.argsort(np.argsort(x[finite])).astype(float)
    rank_y = np.argsort(np.argsort(y[finite])).astype(float)
    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()
    denominator = np.sqrt((rank_x @ rank_x) * (rank_y @ rank_y))
    return float(rank_x @ rank_y / denominator) if denominator > 0 else float("nan")
