# floorbench

An absolute floor for forecasting. Code for "An Absolute Floor for Forecasting, and What It Reveals About Temporal Foundation Models" (NeurIPS 2026 Workshop on Foundation Models for Temporal Systems).

A chaotic system produces information at a rate equal to the sum of its positive Lyapunov exponents (the Kolmogorov-Sinai entropy, by Pesin's identity). That is the minimum rate of corrections a predictor must receive to stay locked to a trajectory. This repo scores forecasters by the correction stream they need, in nats per unit time, and reports the excess over that known floor. Sweeping the tracking precision separates error caused by the world from error caused by the model.

## How it works

1. Integrate a chaotic system and compute its Lyapunov spectrum, which gives the floor.
2. Run a predictor on its own corrected output. At each step the coder sends the integer `n = round((x - prediction) / precision)`, and the predictor only ever sees `prediction + n * precision`.
3. Estimate the entropy rate of the correction stream with an adaptive order-3 context model. Score is this rate minus the floor.
4. Repeat at five precisions. A flat or negative slope of the score against `log2(1/precision)` means the predictor tracks the dynamics. A rising slope means its own error dominates.

Forecasts are consumed as 32-step blocks and only the rows that received a correction are re-forecast, so compute scales with corrections rather than timesteps.

## Setup

```
python -m venv .venv
pip install -r requirements.txt
```

Install a CUDA build of torch first if you want GPU for the foundation models. The delay embedding and persistence baselines run on CPU.

## Run

```
pytest
python scripts/run_benchmark.py --skip-foundation
python scripts/run_benchmark.py --foundation chronos-bolt-tiny chronos-bolt-mini chronos-bolt-small chronos-bolt-base
```

Options: `--systems` to pick systems (van_der_pol, rossler, halvorsen, lorenz, chen), `--horizon`, `--config configs/default.yaml`. Outputs go to `results/benchmark/` as `tables/measurements.csv`, `tables/leaderboard.csv` and figures. Each run overwrites `measurements.csv`, so copy it before launching another. The full paper run took roughly 50 minutes on one consumer GPU.

## Reproduce the paper figures and tables

The final measurements are included, so no rerun is needed.

```
python scripts/make_paper_assets.py
```

This reads `data/measurements_all.csv` and writes the figures, LaTeX tables and headline numbers to `results/paper_assets/`. Pass a different CSV path as the first argument to use your own run.

## Results

| Finding | Value |
| --- | --- |
| Persistence beats every Chronos-Bolt size | 5 of 5 systems |
| Chronos-Bolt precision slopes | inside the persistence band on all systems |
| Delay embedding excess on van der Pol (floor 0) | 0.001 nats per unit time |
| Pooled scaling slope | -11.46 nats per decade of parameters |

The scaling extrapolation uses four sizes from one model family and is an order-of-magnitude illustration only.

## Layout

```
floorbench/
  systems/      dynamics, RK4 integration, Lyapunov spectra, datasets
  predictors/   persistence, delay embedding, Chronos-Bolt adapter
  encoding.py   event-driven predictive coder and precision slope
  entropy.py    context-model entropy rate estimator
scripts/        run_benchmark.py, make_paper_assets.py
data/           final measurements from the paper
paper/          LaTeX source
tests/
```

## Scope

Five low-dimensional systems, one foundation model family (Chronos-Bolt), univariate adapters run per channel. The delay embedding is fitted on extra trajectories and is a calibration reference, not a like-for-like comparison. The measured rate includes coder overhead, so the excess is an upper bound on model deficiency.

## Citation

```
@inproceedings{chaudhry2026floor,
  title  = {An Absolute Floor for Forecasting, and What It Reveals About Temporal Foundation Models},
  author = {Chaudhry, Sarim},
  booktitle = {NeurIPS 2026 Workshop on Foundation Models for Temporal Systems},
  year   = {2026}
}
```

## License

MIT
