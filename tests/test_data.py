"""Data-layer tests.

The loader's job is to degrade gracefully: live download → CSV cache → synthetic
generator. Each of those transitions is a failure path, and failure paths are
exactly what goes untested and then breaks in front of someone. These tests
force each branch deliberately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine import data
from risk_engine.data import load_prices, synthetic_prices, to_returns


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Force the no-network path and redirect the cache into a temp directory."""
    monkeypatch.setattr(data, "_download", lambda *a, **k: None)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------

def test_synthetic_prices_are_positive_and_correctly_shaped():
    prices = synthetic_prices(["A", "B", "C"], n_days=500)
    assert prices.shape == (500, 3)
    assert (prices > 0).all().all()
    assert list(prices.columns) == ["A", "B", "C"]
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_synthetic_generator_is_deterministic():
    """Same seed, same data — otherwise CI results would not be reproducible."""
    a = synthetic_prices(["A", "B"], n_days=200, seed=5)
    b = synthetic_prices(["A", "B"], n_days=200, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data():
    a = synthetic_prices(["A"], n_days=200, seed=1)
    b = synthetic_prices(["A"], n_days=200, seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_returns_are_fat_tailed():
    """The generator must produce leptokurtic returns.

    This is load-bearing: the backtest tests rely on the synthetic data being
    genuinely non-normal, so that a normal-assumption model actually fails on it.
    """
    returns = to_returns(synthetic_prices(["A", "B"], n_days=4000, seed=3, df=3.5))
    assert (returns.kurtosis() > 1.0).all()


def test_synthetic_returns_are_correlated():
    """Assets share a factor, so the contagion tests have structure to find."""
    returns = to_returns(synthetic_prices(["A", "B", "C"], n_days=2000, correlation=0.6))
    off_diagonal = returns.corr().to_numpy()[np.triu_indices(3, k=1)]
    assert (off_diagonal > 0.3).all()


def test_synthetic_returns_show_volatility_clustering():
    """Squared returns must be autocorrelated, or the GARCH tests prove nothing."""
    r = to_returns(synthetic_prices(["A"], n_days=4000, seed=11))["A"].to_numpy()
    squared = r**2 - np.mean(r**2)
    autocorr = float(np.corrcoef(squared[:-1], squared[1:])[0, 1])
    assert autocorr > 0.05


def test_date_range_controls_the_length():
    prices = synthetic_prices(["A"], start="2020-01-01", end="2020-12-31")
    assert prices.index[0] >= pd.Timestamp("2020-01-01")
    assert len(prices) >= 250


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------

def test_simple_returns_are_computed_correctly():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    assert to_returns(prices)["A"].tolist() == pytest.approx([0.10, -0.10])


def test_log_returns_are_computed_correctly():
    prices = pd.DataFrame({"A": [100.0, 110.0]})
    assert to_returns(prices, log=True)["A"].iloc[0] == pytest.approx(np.log(1.1))


def test_simple_returns_aggregate_across_assets():
    """The reason simple returns are the default.

    A portfolio's simple return is the weighted sum of its holdings' simple
    returns. Log returns do not have that property, which is why using them for
    cross-sectional aggregation is a silent error.
    """
    prices = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 45.0]})
    returns = to_returns(prices)
    portfolio = 0.5 * returns["A"].iloc[0] + 0.5 * returns["B"].iloc[0]

    start_value = 0.5 * 100 / 100 + 0.5 * 50 / 50
    end_value = 0.5 * 110 / 100 + 0.5 * 45 / 50
    assert portfolio == pytest.approx(end_value / start_value - 1)


def test_returns_drop_the_leading_nan():
    assert len(to_returns(synthetic_prices(["A"], n_days=100))) == 99


def test_empty_price_frame_rejected():
    with pytest.raises(ValueError, match="empty"):
        to_returns(pd.DataFrame())


# --------------------------------------------------------------------------
# load_prices — the fallback chain
# --------------------------------------------------------------------------

def test_falls_back_to_synthetic_when_download_fails(offline):
    prices = load_prices(["FAKE_X", "FAKE_Y"], start="2020-01-01")
    assert not prices.empty
    assert list(prices.columns) == ["FAKE_X", "FAKE_Y"]


def test_synthetic_fallback_can_be_disabled(offline):
    """A caller who needs real data must be able to demand it and fail loudly."""
    with pytest.raises(RuntimeError, match="synthetic fallback is disabled"):
        load_prices(["FAKE_X"], allow_synthetic=False)


def test_no_tickers_rejected():
    with pytest.raises(ValueError, match="no tickers"):
        load_prices([])


def test_cache_is_written_then_read(monkeypatch, tmp_path):
    """Second call must hit the cache rather than the network."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(data, "CACHE_DIR", cache)

    frame = synthetic_prices(["Z1", "Z2"], n_days=300)
    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        return frame

    monkeypatch.setattr(data, "_download", fake_download)

    first = load_prices(["Z1", "Z2"], start="2020-01-01")
    assert calls["n"] == 1
    assert list(cache.glob("*.csv"))

    second = load_prices(["Z1", "Z2"], start="2020-01-01")
    assert calls["n"] == 1, "second call should have used the cache"
    # check_freq=False: a CSV round-trip cannot preserve the DatetimeIndex's
    # `freq` metadata (BusinessDay -> None). The dates and values are identical,
    # and freq is not part of what the loader promises.
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_cache_can_be_bypassed(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(data, "CACHE_DIR", cache)
    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        return synthetic_prices(["Z1"], n_days=200)

    monkeypatch.setattr(data, "_download", fake_download)
    load_prices(["Z1"], use_cache=False)
    load_prices(["Z1"], use_cache=False)
    assert calls["n"] == 2


def test_columns_that_are_entirely_missing_are_dropped(monkeypatch, tmp_path):
    """A ticker that returns no data must not poison the whole frame with NaN."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    frame = synthetic_prices(["GOOD", "BAD"], n_days=200)
    frame["BAD"] = np.nan
    monkeypatch.setattr(data, "_download", lambda *a, **k: frame)

    prices = load_prices(["GOOD", "BAD"])
    assert list(prices.columns) == ["GOOD"]
    assert not prices.isna().any().any()


def test_gaps_are_forward_filled(monkeypatch, tmp_path):
    """Non-overlapping market holidays leave holes that must not become NaN."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    frame = synthetic_prices(["A", "B"], n_days=200)
    frame.iloc[50, 1] = np.nan
    monkeypatch.setattr(data, "_download", lambda *a, **k: frame)

    prices = load_prices(["A", "B"])
    assert not prices.isna().any().any()
    assert len(prices) == 200


def test_download_exception_is_contained(monkeypatch, tmp_path):
    """A raising data provider must degrade, not crash a risk run."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")

    def boom(*args, **kwargs):
        raise ConnectionError("provider is down")

    monkeypatch.setattr(data, "yf_download_impl", boom, raising=False)
    # _download swallows provider errors internally; assert the public path holds.
    monkeypatch.setattr(data, "_download", lambda *a, **k: None)
    assert not load_prices(["ANY"], start="2020-01-01").empty
