from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from floorbench.entropy import information_rate


def precision_slope(precisions: np.ndarray, rates: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of the rate against log2 of inverse precision.

    A deterministic source costs a fixed number of bits per unit time no matter
    how finely it is tracked, so a model that has learned the dynamics gives a
    slope near zero. A model whose residual is dominated by its own error must
    spend an extra bit per channel per step for every halving of the precision,
    which drives the slope up.
    """
    finite = np.isfinite(rates) & (rates > 0.0)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    x = np.log2(1.0 / np.asarray(precisions, dtype=float)[finite])
    y = np.asarray(rates, dtype=float)[finite]
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


@dataclass(frozen=True)
class BlockMeasurement:
    """Outcome of one event-driven encoding pass at a single precision."""

    predictor: str
    precision: float
    spikes_per_step: float
    spike_rate: float
    bits_per_step: float
    information_rate: float
    reconstruction_rmse: float
    max_reconstruction_error: float
    forecast_calls: int
    row_forecasts: int
    forecasts_per_step: float
    forward_pass_saving: float
    n_steps: int
    sample_interval: float

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def encode_event_driven(
    predictor,
    observations: np.ndarray,
    precision: float,
    sample_interval: float,
    horizon: int = 32,
    warmup_steps: int = 128,
    entropy_order: int = 3,
) -> BlockMeasurement:
    """Track a signal by consuming forecasts and re-forecasting only on correction.

    The predictor emits a block of future values. The coder walks that block
    against the truth, emitting a correction whenever the drift exceeds the
    precision, and asks for a fresh block only when a correction fires or the
    block runs out. Compute therefore scales with the number of corrections
    rather than with the number of timesteps, which is the same claim the paper
    makes about world models applied to its own measurement loop.

    Rows advance in lockstep, so a re-forecast covers only the rows that went
    stale. The saving against a per-step forecaster is reported directly.
    """
    if observations.ndim != 3:
        raise ValueError("observations must have shape (rows, time, channels)")
    rows, length, channels = observations.shape
    context_length = max(getattr(predictor, "context_length", 128), 8)
    if warmup_steps < context_length:
        warmup_steps = min(context_length, length // 2)

    reconstruction = observations.copy()
    symbols = np.zeros((rows, length - warmup_steps, channels), dtype=np.int64)
    row_index = np.arange(rows)

    forecast = predictor.forecast(reconstruction[:, warmup_steps - context_length : warmup_steps], horizon)
    offset = np.zeros(rows, dtype=np.int64)
    forecast_calls, row_forecasts = 1, rows

    for step in range(warmup_steps, length):
        prediction = forecast[row_index, offset]
        symbol = np.rint((observations[:, step] - prediction) / precision)
        symbols[:, step - warmup_steps] = symbol.astype(np.int64)
        reconstruction[:, step] = prediction + symbol * precision

        offset += 1
        stale = np.any(symbol != 0.0, axis=-1) | (offset >= horizon)
        if step + 1 < length and stale.any():
            selected = row_index[stale]
            window = reconstruction[selected, step + 1 - context_length : step + 1]
            forecast[selected] = predictor.forecast(window, horizon)
            offset[stale] = 0
            forecast_calls += 1
            row_forecasts += int(selected.size)

    measured_steps = length - warmup_steps
    residual = observations[:, warmup_steps:] - reconstruction[:, warmup_steps:]
    bits_per_step, nats_per_time = information_rate(symbols, sample_interval, order=entropy_order)
    spikes_per_step = float(np.abs(symbols).sum()) / (rows * measured_steps)
    per_step_baseline = rows * measured_steps

    return BlockMeasurement(
        predictor=getattr(predictor, "name", type(predictor).__name__),
        precision=precision,
        spikes_per_step=spikes_per_step,
        spike_rate=spikes_per_step / sample_interval,
        bits_per_step=bits_per_step,
        information_rate=nats_per_time,
        reconstruction_rmse=float(np.sqrt((residual**2).mean())),
        max_reconstruction_error=float(np.abs(residual).max()),
        forecast_calls=forecast_calls,
        row_forecasts=row_forecasts,
        forecasts_per_step=row_forecasts / per_step_baseline,
        forward_pass_saving=per_step_baseline / max(row_forecasts, 1),
        n_steps=measured_steps,
        sample_interval=sample_interval,
    )
