from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

NATS_PER_BIT = math.log(2.0)


class ContextEntropyCoder:
    """Adaptive order-k context model with Krichevsky-Trofimov smoothing.

    The correction stream is sparse and strongly time-correlated, because a
    correction makes the next one less likely for a while. A marginal symbol
    histogram ignores that structure and badly overstates the rate, and a
    byte-oriented general compressor underperforms on sparse integer streams, so
    the rate is estimated with an explicit context model instead.
    """

    def __init__(self, order: int = 3, alphabet_size: int = 3) -> None:
        if order < 0:
            raise ValueError("order must be non-negative")
        self.order = order
        self.alphabet_size = max(2, alphabet_size)
        self._counts: dict[tuple[int, ...], dict[int, int]] = defaultdict(dict)
        self._totals: dict[tuple[int, ...], int] = defaultdict(int)

    def code_length_bits(self, streams: list[list[int]]) -> float:
        """Total ideal code length in bits for a set of symbol sequences."""
        total_bits = 0.0
        half_alphabet = 0.5 * self.alphabet_size
        for stream in streams:
            context = (0,) * self.order
            for symbol in stream:
                counts = self._counts[context]
                seen = self._totals[context]
                probability = (counts.get(symbol, 0) + 0.5) / (seen + half_alphabet)
                total_bits -= math.log2(probability)
                counts[symbol] = counts.get(symbol, 0) + 1
                self._totals[context] = seen + 1
                if self.order:
                    context = (context + (int(np.sign(symbol)),))[1:]
        return total_bits


def information_rate(
    symbols: np.ndarray, sample_interval: float, order: int = 3
) -> tuple[float, float]:
    """Estimate the correction stream's rate in bits per step and nats per unit time.

    Args:
        symbols: integer corrections, shape (batch, time, channels).
        sample_interval: seconds between samples in the system's own time units.
        order: context length of the adaptive model.
    """
    batch, length, channels = symbols.shape
    alphabet = int(symbols.max() - symbols.min()) + 1
    coder = ContextEntropyCoder(order=order, alphabet_size=alphabet)
    streams = [symbols[b, :, c].tolist() for b in range(batch) for c in range(channels)]
    bits = coder.code_length_bits(streams)
    bits_per_step = bits / (batch * length)
    return bits_per_step, bits_per_step * NATS_PER_BIT / sample_interval


def marginal_entropy_rate(symbols: np.ndarray, sample_interval: float) -> float:
    """Memoryless upper bound on the rate, in nats per unit time."""
    values, counts = np.unique(symbols, return_counts=True)
    probabilities = counts / counts.sum()
    entropy_bits = float(-(probabilities * np.log2(probabilities)).sum())
    return entropy_bits * symbols.shape[-1] * NATS_PER_BIT / sample_interval
