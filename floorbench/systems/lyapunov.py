from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from floorbench.systems.dynamics import DynamicalSystem
from floorbench.systems.integrate import rk4_step, rk4_step_with_tangent


@dataclass(frozen=True)
class LyapunovSpectrum:
    system: str
    exponents: np.ndarray
    kolmogorov_sinai_entropy: float
    largest: float
    lyapunov_time: float
    integration_step: float
    n_steps: int

    def as_record(self) -> dict[str, float | str]:
        return {
            "system": self.system,
            "exponents": ",".join(f"{value:.6f}" for value in self.exponents),
            "ks_entropy": self.kolmogorov_sinai_entropy,
            "largest_exponent": self.largest,
            "lyapunov_time": self.lyapunov_time,
        }


def lyapunov_spectrum(
    system: DynamicalSystem,
    n_steps: int = 60000,
    transient: int = 10000,
    renorm_every: int = 10,
    step: float | None = None,
) -> LyapunovSpectrum:
    """Benettin's algorithm with periodic QR reorthonormalisation.

    Exponents are returned in nats per unit time. The Kolmogorov-Sinai entropy is
    the sum of the positive exponents by Pesin's identity, which is the quantity
    the spike rate is predicted to match.
    """
    step = system.base_step if step is None else step
    state = system.initial_state.copy()
    for _ in range(transient):
        state = rk4_step(system, state, step)

    tangent = np.eye(system.dimension)
    log_growth = np.zeros(system.dimension)
    elapsed = 0.0

    for index in range(n_steps):
        state, tangent = rk4_step_with_tangent(system, state, tangent, step)
        elapsed += step
        if (index + 1) % renorm_every == 0:
            tangent, upper = np.linalg.qr(tangent)
            diagonal = np.abs(np.diag(upper))
            log_growth += np.log(np.maximum(diagonal, 1e-300))
            signs = np.sign(np.diag(upper))
            signs[signs == 0.0] = 1.0
            tangent = tangent * signs

    exponents = np.sort(log_growth / elapsed)[::-1]
    exponents = _zero_neutral_direction(exponents)
    positive = exponents[exponents > 0.0]
    entropy = float(positive.sum())
    largest = float(exponents[0])
    lyapunov_time = float(1.0 / largest) if largest > 1e-6 else float("inf")
    return LyapunovSpectrum(
        system=system.name,
        exponents=exponents,
        kolmogorov_sinai_entropy=entropy,
        largest=largest,
        lyapunov_time=lyapunov_time,
        integration_step=step,
        n_steps=n_steps,
    )


def _zero_neutral_direction(exponents: np.ndarray) -> np.ndarray:
    """An autonomous flow on a non-fixed-point attractor has exactly one zero exponent.

    Finite-time estimation leaves it slightly off zero, which would otherwise leak
    into the Kolmogorov-Sinai sum, so it is pinned to zero explicitly.
    """
    adjusted = exponents.copy()
    adjusted[int(np.argmin(np.abs(adjusted)))] = 0.0
    return adjusted
