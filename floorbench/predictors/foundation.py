from __future__ import annotations

import numpy as np

from floorbench.predictors.base import Predictor, PredictorUnavailable, check_forecast_shape

_INSTALL_HINT = {"chronos": "pip install chronos-forecasting"}


class UnivariateAdapter(Predictor):
    """Runs a univariate backbone once per channel and reassembles the block.

    Most released temporal foundation models are univariate. Treating each
    channel independently is therefore the faithful evaluation, and the
    cross-channel structure the model cannot use is part of the measured gap
    rather than something to hide.
    """

    univariate = True

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        rows, length, channels = context.shape
        flat = context.transpose(0, 2, 1).reshape(rows * channels, length)
        predicted = self._forecast_univariate(flat, horizon)
        forecast = predicted.reshape(rows, channels, horizon).transpose(0, 2, 1)
        return check_forecast_shape(forecast, rows, horizon, channels)

    def _forecast_univariate(self, series: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError


class ChronosPredictor(UnivariateAdapter):
    """Amazon Chronos and Chronos-Bolt through the released pipeline."""

    def __init__(self, checkpoint: str = "amazon/chronos-bolt-small", device: str = "cuda", context_length: int = 512):
        self.name = checkpoint.split("/")[-1]
        self.checkpoint = checkpoint
        self.context_length = context_length
        try:
            import torch
            from chronos import BaseChronosPipeline
        except ImportError as error:
            raise PredictorUnavailable(f"{checkpoint} needs {_INSTALL_HINT['chronos']}") from error
        self._torch = torch
        self._pipeline = BaseChronosPipeline.from_pretrained(checkpoint, device_map=device, torch_dtype=torch.float32)

    def _forecast_univariate(self, series: np.ndarray, horizon: int) -> np.ndarray:
        context = self._torch.from_numpy(series.astype(np.float32))
        _, mean = self._pipeline.predict_quantiles(context, horizon, [0.5])
        return mean.detach().cpu().numpy()[:, :horizon]


REGISTRY = {
    "chronos-bolt-tiny": lambda **kw: ChronosPredictor("amazon/chronos-bolt-tiny", **kw),
    "chronos-bolt-mini": lambda **kw: ChronosPredictor("amazon/chronos-bolt-mini", **kw),
    "chronos-bolt-small": lambda **kw: ChronosPredictor("amazon/chronos-bolt-small", **kw),
    "chronos-bolt-base": lambda **kw: ChronosPredictor("amazon/chronos-bolt-base", **kw),
}

SCALING_FAMILY = ["chronos-bolt-tiny", "chronos-bolt-mini", "chronos-bolt-small", "chronos-bolt-base"]


def build(name: str, **kwargs) -> Predictor:
    if name not in REGISTRY:
        raise KeyError(f"unknown foundation model {name!r}, available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def available(names: list[str], **kwargs) -> tuple[list[Predictor], dict[str, str]]:
    """Build every requested model, reporting rather than raising on missing ones."""
    built, skipped = [], {}
    for name in names:
        try:
            built.append(build(name, **kwargs))
        except (PredictorUnavailable, Exception) as error:  # noqa: BLE001
            skipped[name] = str(error).splitlines()[0][:200]
    return built, skipped
