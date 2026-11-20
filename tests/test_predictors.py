import numpy as np
import pytest

from floorbench.encoding import encode_event_driven, precision_slope
from floorbench.predictors import DelayEmbeddingPredictor, PersistencePredictor
from floorbench.predictors import foundation
from floorbench.predictors.base import PredictorUnavailable, check_forecast_shape


@pytest.fixture
def trajectories():
    time = np.linspace(0.0, 40.0, 1200)
    signal = np.stack([np.sin(time), np.cos(1.3 * time), np.sin(0.7 * time)], axis=-1)
    return np.repeat(signal[None], 4, axis=0).astype(np.float32)


def test_persistence_repeats_the_last_value(trajectories):
    forecast = PersistencePredictor().forecast(trajectories[:, :64], 8)
    assert forecast.shape == (4, 8, 3)
    assert np.allclose(forecast, trajectories[:, 63:64], atol=1e-6)


def test_delay_embedding_beats_persistence_on_a_smooth_signal(trajectories):
    predictor = DelayEmbeddingPredictor(embedding_dim=4, lag=3).fit(trajectories, horizon=8)
    context, truth = trajectories[:, :200], trajectories[:, 200:208]
    embedded = np.sqrt(((predictor.forecast(context, 8) - truth) ** 2).mean())
    naive = np.sqrt(((PersistencePredictor().forecast(context, 8) - truth) ** 2).mean())
    assert embedded < naive


def test_delay_embedding_rejects_a_horizon_it_was_not_fitted_for(trajectories):
    predictor = DelayEmbeddingPredictor().fit(trajectories, horizon=4)
    with pytest.raises(ValueError):
        predictor.forecast(trajectories[:, :64], 16)


def test_check_forecast_shape_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        check_forecast_shape(np.zeros((2, 3, 4)), 2, 5, 4)


@pytest.mark.parametrize("precision", [0.4, 0.1])
def test_event_driven_encoder_respects_the_quantiser_bound(trajectories, precision):
    predictor = DelayEmbeddingPredictor(embedding_dim=4, lag=3).fit(trajectories, horizon=16)
    result = encode_event_driven(predictor, trajectories, precision, 0.05, horizon=16, warmup_steps=64)
    assert result.max_reconstruction_error <= precision / 2 + 1e-5


def test_event_driven_encoder_saves_forward_passes(trajectories):
    predictor = DelayEmbeddingPredictor(embedding_dim=4, lag=3).fit(trajectories, horizon=16)
    result = encode_event_driven(predictor, trajectories, 0.4, 0.05, horizon=16, warmup_steps=64)
    assert result.forward_pass_saving > 1.0
    assert result.forecasts_per_step < 1.0


def test_a_poor_predictor_shows_the_precision_slope(trajectories):
    naive = PersistencePredictor()
    rates = [
        encode_event_driven(naive, trajectories, precision, 0.05, horizon=16, warmup_steps=64).information_rate
        for precision in (0.4, 0.1, 0.025)
    ]
    assert rates[-1] > rates[0]


def test_unavailable_checkpoints_are_reported_not_raised():
    built, skipped = foundation.available(["chronos-bolt-small"])
    assert isinstance(built, list) and isinstance(skipped, dict)
    assert len(built) + len(skipped) == 1


def test_unknown_foundation_model_raises():
    with pytest.raises(KeyError):
        foundation.build("not-a-real-model")


def test_flat_rate_gives_zero_precision_slope():
    slope, _ = precision_slope(np.array([0.8, 0.4, 0.2, 0.1]), np.array([2.0, 2.0, 2.0, 2.0]))
    assert slope == pytest.approx(0.0, abs=1e-9)
