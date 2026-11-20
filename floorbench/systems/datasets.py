from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from floorbench.config import CACHE_ROOT, SystemConfig
from floorbench.systems.dynamics import DynamicalSystem, get_system
from floorbench.systems.integrate import dominant_period, integrate, perturbed_initial_states
from floorbench.systems.lyapunov import LyapunovSpectrum, lyapunov_spectrum


@dataclass
class TrajectoryDataset:
    """Standardised trajectories for one system, plus everything the law needs.

    Observations are z-scored per channel using training statistics, so the
    encoder threshold is expressed in standard deviations and is comparable
    across systems. Rates are reported in the system's native time units.
    """

    system: str
    train: np.ndarray
    evaluation: np.ndarray
    sample_interval: float
    spectrum: LyapunovSpectrum
    channel_mean: np.ndarray
    channel_std: np.ndarray

    @property
    def observation_dim(self) -> int:
        return self.train.shape[-1]

    @property
    def ks_entropy(self) -> float:
        return self.spectrum.kolmogorov_sinai_entropy

    @property
    def steps_per_lyapunov_time(self) -> float:
        if not np.isfinite(self.spectrum.lyapunov_time):
            return float("inf")
        return self.spectrum.lyapunov_time / self.sample_interval


def choose_sample_interval(
    system: DynamicalSystem, spectrum: LyapunovSpectrum, probe: np.ndarray, config: SystemConfig
) -> float:
    """Pick a sampling interval that resolves the system's own timescales.

    Systems differ in natural speed by orders of magnitude, so a shared interval
    would confound the spike rate with an arbitrary unit choice. We target a
    fixed number of samples per Lyapunov time, bounded by the dominant period so
    that non-chaotic systems remain well sampled.
    """
    period = dominant_period(probe, system.base_step)
    period_bound = period / config.points_per_dominant_period
    if spectrum.largest > 1e-6:
        chaos_bound = spectrum.lyapunov_time / config.samples_per_lyapunov_time
        target = min(chaos_bound, period_bound)
    else:
        target = period_bound
    stride = max(1, int(round(target / system.base_step)))
    return stride * system.base_step


def build_dataset(name: str, config: SystemConfig, seed: int = 0, use_cache: bool = True) -> TrajectoryDataset:
    cache_path = _cache_path(name, config, seed)
    if use_cache and cache_path.exists():
        return _load_cache(cache_path, name)

    system = get_system(name)
    spectrum = lyapunov_spectrum(
        system,
        n_steps=config.lyapunov_steps,
        transient=config.lyapunov_transient,
        renorm_every=config.lyapunov_renorm_every,
    )
    probe = integrate(system, 8192, system.base_step, transient=config.transient_length)
    sample_interval = choose_sample_interval(system, spectrum, probe, config)
    stride = max(1, int(round(sample_interval / system.base_step)))

    rng = np.random.default_rng(seed)
    total = config.n_train_trajectories + config.n_eval_trajectories
    starts = perturbed_initial_states(system, total, scale=0.05, rng=rng)
    raw_steps = config.trajectory_length * stride

    trajectories = np.stack(
        [
            integrate(system, raw_steps, system.base_step, transient=config.transient_length, initial_state=start)[
                ::stride
            ]
            for start in starts
        ]
    )

    train = trajectories[: config.n_train_trajectories]
    evaluation = trajectories[config.n_train_trajectories :]
    mean = train.reshape(-1, system.dimension).mean(axis=0)
    std = train.reshape(-1, system.dimension).std(axis=0)
    std = np.maximum(std, 1e-8)

    dataset = TrajectoryDataset(
        system=name,
        train=((train - mean) / std).astype(np.float32),
        evaluation=((evaluation - mean) / std).astype(np.float32),
        sample_interval=sample_interval,
        spectrum=spectrum,
        channel_mean=mean,
        channel_std=std,
    )
    if use_cache:
        _save_cache(cache_path, dataset)
    return dataset


def _cache_path(name: str, config: SystemConfig, seed: int) -> Path:
    payload = {
        "seed": seed,
        "trajectory_length": config.trajectory_length,
        "n_train": config.n_train_trajectories,
        "n_eval": config.n_eval_trajectories,
        "transient": config.transient_length,
        "lyapunov_steps": config.lyapunov_steps,
        "samples_per_lyapunov_time": config.samples_per_lyapunov_time,
        "points_per_dominant_period": config.points_per_dominant_period,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return CACHE_ROOT / "systems" / f"{name}_{digest}.npz"


def _save_cache(path: Path, dataset: TrajectoryDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train=dataset.train,
        evaluation=dataset.evaluation,
        sample_interval=dataset.sample_interval,
        channel_mean=dataset.channel_mean,
        channel_std=dataset.channel_std,
        exponents=dataset.spectrum.exponents,
        ks_entropy=dataset.spectrum.kolmogorov_sinai_entropy,
        largest=dataset.spectrum.largest,
        lyapunov_time=dataset.spectrum.lyapunov_time,
        integration_step=dataset.spectrum.integration_step,
        n_steps=dataset.spectrum.n_steps,
    )


def _load_cache(path: Path, name: str) -> TrajectoryDataset:
    blob = np.load(path)
    spectrum = LyapunovSpectrum(
        system=name,
        exponents=blob["exponents"],
        kolmogorov_sinai_entropy=float(blob["ks_entropy"]),
        largest=float(blob["largest"]),
        lyapunov_time=float(blob["lyapunov_time"]),
        integration_step=float(blob["integration_step"]),
        n_steps=int(blob["n_steps"]),
    )
    return TrajectoryDataset(
        system=name,
        train=blob["train"],
        evaluation=blob["evaluation"],
        sample_interval=float(blob["sample_interval"]),
        spectrum=spectrum,
        channel_mean=blob["channel_mean"],
        channel_std=blob["channel_std"],
    )
