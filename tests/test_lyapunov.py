import numpy as np
import pytest

from floorbench.systems.dynamics import all_system_names, get_system
from floorbench.systems.lyapunov import lyapunov_spectrum

PUBLISHED_LARGEST = {"lorenz": 0.906, "rossler": 0.071}


@pytest.mark.parametrize("name,expected", PUBLISHED_LARGEST.items())
def test_largest_exponent_matches_published_value(name, expected):
    spectrum = lyapunov_spectrum(get_system(name), n_steps=40000, transient=8000)
    assert spectrum.largest == pytest.approx(expected, abs=0.05)


def test_flow_has_one_zero_exponent():
    spectrum = lyapunov_spectrum(get_system("lorenz"), n_steps=20000, transient=5000)
    assert np.isclose(np.min(np.abs(spectrum.exponents)), 0.0)


def test_limit_cycle_has_no_positive_exponent():
    spectrum = lyapunov_spectrum(get_system("van_der_pol"), n_steps=20000, transient=5000)
    assert spectrum.kolmogorov_sinai_entropy == pytest.approx(0.0, abs=1e-3)


def test_hyperchaotic_system_has_two_positive_exponents():
    spectrum = lyapunov_spectrum(get_system("hyper_rossler"), n_steps=40000, transient=8000)
    assert int((spectrum.exponents > 1e-3).sum()) >= 2


def test_every_registered_system_integrates():
    for name in all_system_names():
        spectrum = lyapunov_spectrum(get_system(name), n_steps=4000, transient=1000)
        assert np.all(np.isfinite(spectrum.exponents))
