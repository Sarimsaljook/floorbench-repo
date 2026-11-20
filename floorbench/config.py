from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
CACHE_ROOT = PROJECT_ROOT / "cache"


@dataclass
class SystemConfig:
    names: list[str] = field(default_factory=list)
    n_train_trajectories: int = 24
    n_eval_trajectories: int = 8
    trajectory_length: int = 4096
    transient_length: int = 2048
    lyapunov_steps: int = 60000
    lyapunov_transient: int = 10000
    lyapunov_renorm_every: int = 10
    samples_per_lyapunov_time: float = 25.0
    points_per_dominant_period: float = 40.0


@dataclass
class EncoderConfig:
    precisions: list[float] = field(default_factory=lambda: [0.8, 0.4, 0.2, 0.1, 0.05])
    warmup_steps: int = 128
    entropy_order: int = 3


@dataclass
class Config:
    seed: int = 0
    device: str = "auto"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    system: SystemConfig = field(default_factory=SystemConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return _build(cls, raw)

    def to_dict(self) -> dict[str, Any]:
        return _unbuild(self)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Build a config tree from plain YAML data.

    Field types are resolved from the default instance rather than from the
    annotation, because postponed annotation evaluation makes the annotation a
    string at runtime.
    """
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for spec in fields(cls):
        if spec.name not in raw:
            continue
        value = raw[spec.name]
        current = getattr(defaults, spec.name)
        if is_dataclass(current) and isinstance(value, dict):
            kwargs[spec.name] = _build(type(current), value)
        else:
            kwargs[spec.name] = value
    return cls(**kwargs)


def _unbuild(obj: Any) -> Any:
    if is_dataclass(obj):
        return {spec.name: _unbuild(getattr(obj, spec.name)) for spec in fields(obj)}
    if isinstance(obj, list):
        return [_unbuild(item) for item in obj]
    return obj
