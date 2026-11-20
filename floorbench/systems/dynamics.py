from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Vector = np.ndarray
Matrix = np.ndarray


@dataclass(frozen=True)
class DynamicalSystem:
    """A continuous-time system with an analytic Jacobian.

    The Jacobian is supplied rather than differenced so that the Lyapunov
    spectrum is exact to integrator precision, which matters because the whole
    experiment compares a measured rate against those exponents.
    """

    name: str
    dimension: int
    vector_field: Callable[[Vector], Vector]
    jacobian: Callable[[Vector], Matrix]
    initial_state: Vector
    base_step: float
    chaotic: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)


def _lorenz(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def jac_fn(s: Vector) -> Matrix:
        x, y, z = s
        return np.array([[-sigma, sigma, 0.0], [rho - z, -1.0, -x], [y, x, -beta]])

    return DynamicalSystem("lorenz", 3, field_fn, jac_fn, np.array([1.0, 1.0, 20.0]), 0.005, tags=("classic",))


def _rossler(a: float = 0.2, b: float = 0.2, c: float = 5.7) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    def jac_fn(s: Vector) -> Matrix:
        x, _, z = s
        return np.array([[0.0, -1.0, -1.0], [1.0, a, 0.0], [z, 0.0, x - c]])

    return DynamicalSystem("rossler", 3, field_fn, jac_fn, np.array([1.0, 1.0, 1.0]), 0.02, tags=("classic",))


def _chen(a: float = 35.0, b: float = 3.0, c: float = 28.0) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array([a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z])

    def jac_fn(s: Vector) -> Matrix:
        x, y, z = s
        return np.array([[-a, a, 0.0], [c - a - z, c, -x], [y, x, -b]])

    return DynamicalSystem("chen", 3, field_fn, jac_fn, np.array([-0.1, 0.5, -0.6]), 0.002)


def _lu(a: float = 36.0, b: float = 3.0, c: float = 20.0) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array([a * (y - x), -x * z + c * y, x * y - b * z])

    def jac_fn(s: Vector) -> Matrix:
        x, y, z = s
        return np.array([[-a, a, 0.0], [-z, c, -x], [y, x, -b]])

    return DynamicalSystem("lu", 3, field_fn, jac_fn, np.array([0.1, 0.3, -0.6]), 0.002)


def _halvorsen(a: float = 1.4) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array(
            [
                -a * x - 4.0 * y - 4.0 * z - y * y,
                -a * y - 4.0 * z - 4.0 * x - z * z,
                -a * z - 4.0 * x - 4.0 * y - x * x,
            ]
        )

    def jac_fn(s: Vector) -> Matrix:
        x, y, z = s
        return np.array(
            [
                [-a, -4.0 - 2.0 * y, -4.0],
                [-4.0, -a, -4.0 - 2.0 * z],
                [-4.0 - 2.0 * x, -4.0, -a],
            ]
        )

    return DynamicalSystem("halvorsen", 3, field_fn, jac_fn, np.array([-5.0, 0.0, 0.0]), 0.005)


def _thomas(b: float = 0.18) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array([np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z])

    def jac_fn(s: Vector) -> Matrix:
        x, y, z = s
        return np.array([[-b, np.cos(y), 0.0], [0.0, -b, np.cos(z)], [np.cos(x), 0.0, -b]])

    return DynamicalSystem("thomas", 3, field_fn, jac_fn, np.array([0.1, 0.0, 0.0]), 0.02, tags=("slow",))


def _dadras(a: float = 3.0, b: float = 2.7, c: float = 1.7, d: float = 2.0, e: float = 9.0) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array([y - a * x + b * y * z, c * y - x * z + z, d * x * y - e * z])

    def jac_fn(s: Vector) -> Matrix:
        x, y, z = s
        return np.array([[-a, 1.0 + b * z, b * y], [-z, c, -x + 1.0], [d * y, d * x, -e]])

    return DynamicalSystem("dadras", 3, field_fn, jac_fn, np.array([1.1, 2.1, -2.0]), 0.01)


def _four_wing(a: float = 0.2, b: float = 0.01, c: float = -0.4) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array([a * x + y * z, b * x + c * y - x * z, -z - x * y])

    def jac_fn(s: Vector) -> Matrix:
        x, y, z = s
        return np.array([[a, z, y], [b - z, c, -x], [-y, -x, -1.0]])

    return DynamicalSystem("four_wing", 3, field_fn, jac_fn, np.array([1.0, -1.0, 1.0]), 0.02)


def _rikitake(mu: float = 1.0, a: float = 5.0) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array([-mu * x + z * y, -mu * y + x * (z - a), 1.0 - x * y])

    def jac_fn(s: Vector) -> Matrix:
        x, y, z = s
        return np.array([[-mu, z, y], [z - a, -mu, x], [-y, -x, 0.0]])

    return DynamicalSystem("rikitake", 3, field_fn, jac_fn, np.array([1.0, 0.0, 0.6]), 0.01)


def _lorenz84(a: float = 0.25, b: float = 4.0, forcing: float = 8.0, drive: float = 1.0) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array(
            [
                -y * y - z * z - a * x + a * forcing,
                x * y - b * x * z - y + drive,
                b * x * y + x * z - z,
            ]
        )

    def jac_fn(s: Vector) -> Matrix:
        x, y, z = s
        return np.array(
            [
                [-a, -2.0 * y, -2.0 * z],
                [y - b * z, x - 1.0, -b * x],
                [b * y + z, b * x, x - 1.0],
            ]
        )

    return DynamicalSystem("lorenz84", 3, field_fn, jac_fn, np.array([0.5, 0.5, 0.5]), 0.02)


def _sprott_b() -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array([y * z, x - y, 1.0 - x * y])

    def jac_fn(s: Vector) -> Matrix:
        x, y, z = s
        return np.array([[0.0, z, y], [1.0, -1.0, 0.0], [-y, -x, 0.0]])

    return DynamicalSystem("sprott_b", 3, field_fn, jac_fn, np.array([0.1, 0.2, 0.3]), 0.02)


def _chua(alpha: float = 15.6, beta: float = 28.0, m0: float = -1.143, m1: float = -0.714) -> DynamicalSystem:
    def nonlinearity(x: float) -> float:
        return m1 * x + 0.5 * (m0 - m1) * (abs(x + 1.0) - abs(x - 1.0))

    def slope(x: float) -> float:
        return m0 if abs(x) < 1.0 else m1

    def field_fn(s: Vector) -> Vector:
        x, y, z = s
        return np.array([alpha * (y - x - nonlinearity(x)), x - y + z, -beta * y])

    def jac_fn(s: Vector) -> Matrix:
        x, _, _ = s
        return np.array([[alpha * (-1.0 - slope(x)), alpha, 0.0], [1.0, -1.0, 1.0], [0.0, -beta, 0.0]])

    return DynamicalSystem("chua", 3, field_fn, jac_fn, np.array([0.7, 0.0, -0.5]), 0.005)


def _hyper_rossler(a: float = 0.25, b: float = 3.0, c: float = 0.5, d: float = 0.05) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z, w = s
        return np.array([-y - z, x + a * y + w, b + x * z, -c * z + d * w])

    def jac_fn(s: Vector) -> Matrix:
        x, _, z, _ = s
        return np.array(
            [
                [0.0, -1.0, -1.0, 0.0],
                [1.0, a, 0.0, 1.0],
                [z, 0.0, x, 0.0],
                [0.0, 0.0, -c, d],
            ]
        )

    return DynamicalSystem(
        "hyper_rossler", 4, field_fn, jac_fn, np.array([-10.0, -6.0, 0.0, 10.0]), 0.02, tags=("hyperchaotic",)
    )


def _hyper_lorenz(a: float = 10.0, b: float = 8.0 / 3.0, c: float = 28.0, d: float = 1.0) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y, z, w = s
        return np.array([a * (y - x) + w, c * x - y - x * z, x * y - b * z, d * w - x * z])

    def jac_fn(s: Vector) -> Matrix:
        x, y, z, _ = s
        return np.array(
            [
                [-a, a, 0.0, 1.0],
                [c - z, -1.0, -x, 0.0],
                [y, x, -b, 0.0],
                [-z, 0.0, -x, d],
            ]
        )

    return DynamicalSystem(
        "hyper_lorenz", 4, field_fn, jac_fn, np.array([1.0, 1.0, 1.0, 1.0]), 0.002, tags=("hyperchaotic",)
    )


def _van_der_pol(mu: float = 2.0) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y = s
        return np.array([y, mu * (1.0 - x * x) * y - x])

    def jac_fn(s: Vector) -> Matrix:
        x, y = s
        return np.array([[0.0, 1.0], [-2.0 * mu * x * y - 1.0, mu * (1.0 - x * x)]])

    return DynamicalSystem(
        "van_der_pol", 2, field_fn, jac_fn, np.array([2.0, 0.0]), 0.02, chaotic=False, tags=("limit_cycle",)
    )


def _hopf(mu: float = 1.0, omega: float = 1.0) -> DynamicalSystem:
    def field_fn(s: Vector) -> Vector:
        x, y = s
        radius = x * x + y * y
        return np.array([mu * x - omega * y - x * radius, omega * x + mu * y - y * radius])

    def jac_fn(s: Vector) -> Matrix:
        x, y = s
        radius = x * x + y * y
        return np.array(
            [
                [mu - radius - 2.0 * x * x, -omega - 2.0 * x * y],
                [omega - 2.0 * x * y, mu - radius - 2.0 * y * y],
            ]
        )

    return DynamicalSystem(
        "hopf", 2, field_fn, jac_fn, np.array([0.5, 0.0]), 0.02, chaotic=False, tags=("limit_cycle",)
    )


_REGISTRY: dict[str, DynamicalSystem] = {
    system.name: system
    for system in (
        _lorenz(),
        _rossler(),
        _chen(),
        _lu(),
        _halvorsen(),
        _thomas(),
        _dadras(),
        _four_wing(),
        _rikitake(),
        _lorenz84(),
        _sprott_b(),
        _chua(),
        _hyper_rossler(),
        _hyper_lorenz(),
        _van_der_pol(),
        _hopf(),
    )
}


def get_system(name: str) -> DynamicalSystem:
    if name not in _REGISTRY:
        raise KeyError(f"unknown system {name!r}, available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_system_names() -> list[str]:
    return sorted(_REGISTRY)
