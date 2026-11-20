import math

import numpy as np
import pytest

from floorbench.entropy import information_rate, marginal_entropy_rate


def analytic_ternary_entropy(p_nonzero: float) -> float:
    half = p_nonzero / 2.0
    zero = 1.0 - p_nonzero
    return -(zero * math.log2(zero) + 2 * half * math.log2(half))


@pytest.mark.parametrize("p_nonzero", [0.02, 0.1])
def test_rate_matches_analytic_entropy_of_an_iid_stream(p_nonzero):
    rng = np.random.default_rng(0)
    draws = rng.random((4, 4000, 3))
    symbols = (draws < p_nonzero).astype(np.int64) * rng.choice([-1, 1], (4, 4000, 3))
    bits_per_step, _ = information_rate(symbols, sample_interval=1.0)
    expected = analytic_ternary_entropy(p_nonzero) * 3
    assert bits_per_step == pytest.approx(expected, rel=0.1)


def test_all_zero_stream_costs_almost_nothing():
    symbols = np.zeros((2, 2000, 3), dtype=np.int64)
    bits_per_step, _ = information_rate(symbols, sample_interval=1.0)
    assert bits_per_step < 0.02


def test_marginal_rate_upper_bounds_the_context_model():
    rng = np.random.default_rng(1)
    symbols = rng.integers(-2, 3, (2, 2000, 3))
    _, contextual = information_rate(symbols, sample_interval=0.1)
    assert marginal_entropy_rate(symbols, 0.1) >= contextual * 0.95
