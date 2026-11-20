"""The floor-referenced forecasting benchmark.

Every forecasting metric in common use is relative, so nobody knows how far a
model sits from what is achievable. On systems with a computable
Kolmogorov-Sinai entropy the achievable value is known exactly, so this scores
predictors in excess bits per unit time over a known optimum, and reports how
that excess responds to the tracking precision.

Inference only. The delay embedding is the one fitted baseline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from floorbench import plotting
from floorbench.config import Config
from floorbench.encoding import encode_event_driven, precision_slope
from floorbench.metrics import coding_gap, coding_ratio, linear_fit, spearman
from floorbench.predictors import DelayEmbeddingPredictor, PersistencePredictor
from floorbench.predictors import foundation
from floorbench.systems.datasets import build_dataset
from floorbench.systems.dynamics import all_system_names
from floorbench.utils import ResultWriter, enable_determinism, get_logger, resolve_device

LOGGER = get_logger("floorbench")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--foundation", nargs="*", default=list(foundation.REGISTRY))
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--skip-foundation", action="store_true")
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    config = Config.from_yaml(arguments.config)
    systems = arguments.systems or config.system.names or all_system_names()
    device = resolve_device(arguments.device or config.device)
    enable_determinism()
    writer = ResultWriter("benchmark")

    checkpoints, skipped = ([], {})
    if not arguments.skip_foundation:
        checkpoints, skipped = foundation.available(arguments.foundation, device=str(device))
    if skipped:
        LOGGER.warning("unavailable: %s", ", ".join(sorted(skipped)))
        writer.write_json(skipped, "unavailable_checkpoints")
    LOGGER.info("device %s | %d systems | %d foundation models", device, len(systems), len(checkpoints))

    records: list[dict] = []
    started = time.time()

    for system in systems:
        dataset = build_dataset(system, config.system, seed=0)
        predictors = [
            PersistencePredictor(),
            DelayEmbeddingPredictor(embedding_dim=dataset.observation_dim + 1, lag=3).fit(
                dataset.train, horizon=arguments.horizon
            ),
            *checkpoints,
        ]

        for predictor in predictors:
            for precision in config.encoder.precisions:
                measurement = encode_event_driven(
                    predictor,
                    dataset.evaluation,
                    precision=precision,
                    sample_interval=dataset.sample_interval,
                    horizon=arguments.horizon,
                    warmup_steps=config.encoder.warmup_steps,
                    entropy_order=config.encoder.entropy_order,
                )
                records.append(
                    {
                        "system": system,
                        "ks_entropy": dataset.ks_entropy,
                        "largest_exponent": dataset.spectrum.largest,
                        "observation_dim": dataset.observation_dim,
                        "sample_interval": dataset.sample_interval,
                        "steps_per_lyapunov_time": dataset.steps_per_lyapunov_time,
                        **measurement.as_record(),
                    }
                )
            coarse = records[-len(config.encoder.precisions)]
            LOGGER.info(
                "%-14s %-22s excess %8.2f nats/t | %.1fx forward-pass saving",
                system,
                coarse["predictor"],
                coarse["information_rate"] - coarse["ks_entropy"],
                coarse["forward_pass_saving"],
            )

    frame = pd.DataFrame(records)
    frame["coding_gap"] = coding_gap(frame["information_rate"].to_numpy(), frame["ks_entropy"].to_numpy())
    frame["coding_ratio"] = coding_ratio(frame["information_rate"].to_numpy(), frame["ks_entropy"].to_numpy())
    writer.write_table(frame, "measurements")

    leaderboard = _leaderboard(frame)
    writer.write_table(leaderboard, "leaderboard")
    writer.write_json(_law_fits(frame), "law_fits")

    averaged = frame.groupby(["predictor", "precision"], as_index=False).mean(numeric_only=True)
    averaged["system"] = averaged["predictor"]
    plotting.precision_sweep_figure(averaged, writer.figure_path("precision_sweep_by_predictor"))
    for predictor in frame["predictor"].unique():
        subset = frame[frame["predictor"] == predictor]
        plotting.plateau_figure(subset, writer.figure_path(f"floor_{predictor}"))

    LOGGER.info("finished in %.0fs, results in %s", time.time() - started, writer.root)
    LOGGER.info("\n%s", leaderboard.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor, group in frame.groupby("predictor"):
        per_system = group.groupby("system")
        slopes = [
            precision_slope(
                sub.sort_values("precision", ascending=False)["precision"].to_numpy(),
                sub.sort_values("precision", ascending=False)["information_rate"].to_numpy(),
            )[0]
            for _, sub in per_system
        ]
        coarsest = group[group["precision"] == group["precision"].max()]
        rows.append(
            {
                "predictor": predictor,
                "median_excess_bits": float(group["coding_gap"].median()),
                "median_coding_ratio": float(group["coding_ratio"].median()),
                "excess_at_coarsest": float(coarsest["coding_gap"].median()),
                "median_precision_slope": float(np.nanmedian(slopes)),
                "forward_pass_saving": float(group["forward_pass_saving"].median()),
                "n_systems": int(group["system"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("median_excess_bits").reset_index(drop=True)


def _law_fits(frame: pd.DataFrame) -> dict:
    fits = {}
    coarsest = frame["precision"].max()
    for predictor, group in frame[frame["precision"] == coarsest].groupby("predictor"):
        summary = group.groupby("system", as_index=False).mean(numeric_only=True)
        fits[predictor] = {
            "rate_vs_entropy": linear_fit(
                summary["ks_entropy"].to_numpy(), summary["information_rate"].to_numpy()
            ),
            "rank_correlation": spearman(
                summary["ks_entropy"].to_numpy(), summary["information_rate"].to_numpy()
            ),
        }
    return {
        "coarsest_precision": float(coarsest),
        "per_predictor": fits,
        "note": "unit slope through the origin and a zero precision slope would mean the floor is reached",
    }


if __name__ == "__main__":
    main()
