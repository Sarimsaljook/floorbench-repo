from __future__ import annotations

import numpy as np

from floorbench.predictors.base import Predictor, check_forecast_shape


class PersistencePredictor(Predictor):
    """Repeat the last observation. The trivial reference every method must beat."""

    name = "persistence"
    context_length = 8

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        last = context[:, -1:, :]
        return check_forecast_shape(np.repeat(last, horizon, axis=1), *context.shape[:1], horizon, context.shape[2])


class DelayEmbeddingPredictor(Predictor):
    """Local linear analog forecasting on a delay embedding.

    Classical nonlinear time series analysis is very strong on low dimensional
    deterministic dynamics, which is exactly the regime this benchmark measures.
    Including it is the point rather than a formality, since a method with a few
    hundred effective parameters closing more of the gap than a pretrained
    transformer would be the most informative outcome available.
    """

    name = "delay_embedding"

    def __init__(
        self,
        embedding_dim: int = 3,
        lag: int = 1,
        n_neighbours: int = 24,
        library_size: int = 20000,
        ridge: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.lag = lag
        self.n_neighbours = n_neighbours
        self.library_size = library_size
        self.ridge = ridge
        self.seed = seed
        self.context_length = embedding_dim * lag + 1
        self._library_embeddings: np.ndarray | None = None
        self._library_futures: np.ndarray | None = None
        self._horizon = 0

    def fit(self, trajectories: np.ndarray, horizon: int = 32) -> "DelayEmbeddingPredictor":
        span = (self.embedding_dim - 1) * self.lag
        embeddings, futures = [], []
        for trajectory in trajectories:
            usable = trajectory.shape[0] - span - horizon
            if usable <= 0:
                continue
            index = np.arange(usable) + span
            offsets = -np.arange(self.embedding_dim) * self.lag
            embeddings.append(trajectory[index[:, None] + offsets[None, :]].reshape(usable, -1))
            futures.append(np.stack([trajectory[index + step + 1] for step in range(horizon)], axis=1))

        embedding_array = np.concatenate(embeddings, axis=0)
        future_array = np.concatenate(futures, axis=0)
        if embedding_array.shape[0] > self.library_size:
            rng = np.random.default_rng(self.seed)
            keep = rng.choice(embedding_array.shape[0], self.library_size, replace=False)
            embedding_array, future_array = embedding_array[keep], future_array[keep]

        self._library_embeddings = embedding_array.astype(np.float32)
        self._library_futures = future_array.astype(np.float32)
        self._horizon = horizon
        return self

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        if self._library_embeddings is None:
            raise RuntimeError("call fit before forecast")
        if horizon > self._horizon:
            raise ValueError(f"fitted for horizon {self._horizon}, asked for {horizon}")

        rows, _, channels = context.shape
        offsets = -np.arange(self.embedding_dim) * self.lag - 1
        query = context[:, offsets, :].reshape(rows, -1).astype(np.float32)

        distances = _squared_distances(query, self._library_embeddings)
        neighbours = np.argpartition(distances, self.n_neighbours, axis=1)[:, : self.n_neighbours]

        predictions = np.empty((rows, horizon, channels), dtype=np.float32)
        for row in range(rows):
            index = neighbours[row]
            design = np.concatenate(
                [self._library_embeddings[index] - query[row], np.ones((self.n_neighbours, 1), np.float32)], axis=1
            )
            targets = self._library_futures[index][:, :horizon].reshape(self.n_neighbours, -1)
            gram = design.T @ design + self.ridge * np.eye(design.shape[1], dtype=np.float32)
            coefficients = np.linalg.solve(gram, design.T @ targets)
            predictions[row] = coefficients[-1].reshape(horizon, channels)

        return check_forecast_shape(predictions, rows, horizon, channels)

    @property
    def description(self) -> str:
        return f"delay embedding m={self.embedding_dim} lag={self.lag} k={self.n_neighbours}"


def _squared_distances(query: np.ndarray, library: np.ndarray) -> np.ndarray:
    return (
        (query * query).sum(axis=1, keepdims=True)
        - 2.0 * query @ library.T
        + (library * library).sum(axis=1)[None, :]
    )
