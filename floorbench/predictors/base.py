from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Predictor(ABC):
    """A next-block forecaster the coder can drive.

    The floor benchmark compares anything that can extend a context, so the
    interface is deliberately narrow. Everything downstream, meaning the coder,
    the entropy estimator and the leaderboard, is shared across classical
    methods, pretrained foundation models, and models trained in domain.
    """

    name: str = "predictor"
    context_length: int = 256
    univariate: bool = False

    @abstractmethod
    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Extend each row of context by horizon steps.

        Args:
            context: recent reconstructed observations, shape (rows, length, channels).
            horizon: number of steps to predict.

        Returns:
            Forecast of shape (rows, horizon, channels).
        """

    def fit(self, trajectories: np.ndarray) -> "Predictor":
        """Optional in-domain fitting. Zero-shot predictors ignore the data."""
        return self

    @property
    def description(self) -> str:
        return self.name


class PredictorUnavailable(RuntimeError):
    """Raised when an optional checkpoint or dependency is not installed."""


def check_forecast_shape(forecast: np.ndarray, rows: int, horizon: int, channels: int) -> np.ndarray:
    if forecast.shape != (rows, horizon, channels):
        raise ValueError(f"expected forecast shape {(rows, horizon, channels)}, got {forecast.shape}")
    return np.nan_to_num(forecast.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
