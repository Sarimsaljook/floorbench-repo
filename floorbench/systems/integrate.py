from __future__ import annotations

import numpy as np

from floorbench.systems.dynamics import DynamicalSystem, Matrix, Vector


def rk4_step(system: DynamicalSystem, state: Vector, step: float) -> Vector:
    k1 = system.vector_field(state)
    k2 = system.vector_field(state + 0.5 * step * k1)
    k3 = system.vector_field(state + 0.5 * step * k2)
    k4 = system.vector_field(state + step * k3)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step_with_tangent(
    system: DynamicalSystem, state: Vector, tangent: Matrix, step: float
) -> tuple[Vector, Matrix]:
    """Advance the state and its variational equation with the same RK4 tableau."""

    def derivative(s: Vector, t: Matrix) -> tuple[Vector, Matrix]:
        return system.vector_field(s), system.jacobian(s) @ t

    k1_s, k1_t = derivative(state, tangent)
    k2_s, k2_t = derivative(state + 0.5 * step * k1_s, tangent + 0.5 * step * k1_t)
    k3_s, k3_t = derivative(state + 0.5 * step * k2_s, tangent + 0.5 * step * k2_t)
    k4_s, k4_t = derivative(state + step * k3_s, tangent + step * k3_t)
    next_state = state + (step / 6.0) * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)
    next_tangent = tangent + (step / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
    return next_state, next_tangent


def integrate(
    system: DynamicalSystem,
    n_steps: int,
    step: float,
    transient: int = 0,
    initial_state: Vector | None = None,
) -> Vector:
    """Return a trajectory of shape (n_steps, dimension) after discarding a transient."""
    state = system.initial_state.copy() if initial_state is None else np.asarray(initial_state, dtype=float).copy()
    for _ in range(transient):
        state = rk4_step(system, state, step)
    trajectory = np.empty((n_steps, system.dimension), dtype=float)
    for index in range(n_steps):
        trajectory[index] = state
        state = rk4_step(system, state, step)
    if not np.all(np.isfinite(trajectory)):
        raise FloatingPointError(f"{system.name} diverged at step {step}")
    return trajectory


def perturbed_initial_states(
    system: DynamicalSystem, count: int, scale: float, rng: np.random.Generator
) -> list[Vector]:
    spread = np.abs(system.initial_state) + 1.0
    return [system.initial_state + scale * spread * rng.standard_normal(system.dimension) for _ in range(count)]


def dominant_period(trajectory: Vector, step: float) -> float:
    """Period of the strongest non-DC spectral peak, averaged over channels."""
    centred = trajectory - trajectory.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centred, axis=0))
    frequencies = np.fft.rfftfreq(centred.shape[0], d=step)
    spectrum[0] = 0.0
    weights = spectrum.sum(axis=1)
    peak = int(np.argmax(weights))
    if frequencies[peak] <= 0.0:
        return float(centred.shape[0] * step)
    return float(1.0 / frequencies[peak])
