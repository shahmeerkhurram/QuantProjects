"""Backtesting and coverage-test tests.

The coverage tests are themselves statistical procedures, so they are validated
against inputs whose correct verdict is known in advance: a well-calibrated
model must pass, a deliberately broken one must be rejected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.backtest import (
    basel_traffic_light,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    rolling_var_backtest,
)
from risk_engine.data import synthetic_prices, to_returns
from risk_engine.var import portfolio_returns

# --------------------------------------------------------------------------
# Kupiec — unconditional coverage
# --------------------------------------------------------------------------

def test_kupiec_passes_a_perfectly_calibrated_model():
    """Exactly 1% breaches out of 1000 days is the null hypothesis itself."""
    result = kupiec_pof(n_breaches=10, n_obs=1000, confidence=0.99)
    assert result.statistic == pytest.approx(0.0, abs=1e-9)
    assert not result.reject_at_5pct


def test_kupiec_rejects_a_model_that_breaches_far_too_often():
    """50 breaches where 10 were expected: the model badly understates risk."""
    result = kupiec_pof(n_breaches=50, n_obs=1000, confidence=0.99)
    assert result.reject_at_5pct
    assert result.p_value < 0.001
    assert "understating" in result.interpretation


def test_kupiec_rejects_an_over_conservative_model():
    """Zero breaches in 1000 days is also a failure — capital is being wasted."""
    result = kupiec_pof(n_breaches=0, n_obs=1000, confidence=0.99)
    assert result.reject_at_5pct
    assert "overstating" in result.interpretation


def test_kupiec_tolerates_small_deviations():
    """The test must not be so sharp that ordinary sampling noise trips it."""
    assert not kupiec_pof(n_breaches=13, n_obs=1000, confidence=0.99).reject_at_5pct


def test_kupiec_statistic_is_never_negative():
    for x in range(0, 60, 7):
        assert kupiec_pof(x, 1000, 0.99).statistic >= 0.0


# --------------------------------------------------------------------------
# Christoffersen — independence
# --------------------------------------------------------------------------

def test_christoffersen_passes_scattered_breaches():
    """Independent Bernoulli breaches must not be flagged as clustered."""
    rng = np.random.default_rng(42)
    breaches = (rng.random(2000) < 0.01).astype(int)
    assert not christoffersen_independence(breaches).reject_at_5pct


def test_christoffersen_rejects_clustered_breaches():
    """All breaches consecutive — the signature of an unadaptive model."""
    breaches = np.zeros(1000, dtype=int)
    breaches[400:430] = 1
    result = christoffersen_independence(breaches)
    assert result.reject_at_5pct
    assert "cluster" in result.interpretation


def test_christoffersen_handles_zero_breaches():
    """No breaches means no clustering evidence; the test must not divide by zero."""
    result = christoffersen_independence(np.zeros(500, dtype=int))
    assert result.statistic == pytest.approx(0.0)
    assert not result.reject_at_5pct


def test_conditional_coverage_sums_the_components():
    breaches = np.zeros(1000, dtype=int)
    breaches[100:110] = 1
    joint = conditional_coverage(10, 1000, 0.99, breaches)
    pof = kupiec_pof(10, 1000, 0.99)
    ind = christoffersen_independence(breaches)
    assert joint.statistic == pytest.approx(pof.statistic + ind.statistic)
    assert joint.degrees_of_freedom == 2


# --------------------------------------------------------------------------
# Basel traffic light
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "breaches,expected",
    [(0, "GREEN"), (4, "GREEN"), (6, "AMBER"), (9, "AMBER"), (15, "RED")],
)
def test_basel_zones_at_the_supervisory_window(breaches, expected):
    assert basel_traffic_light(breaches, 250, 0.99).startswith(expected)


# --------------------------------------------------------------------------
# Walk-forward backtest
# --------------------------------------------------------------------------

@pytest.fixture
def normal_series() -> pd.Series:
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2015-01-01", periods=3000)
    return pd.Series(rng.normal(0.0003, 0.01, 3000), index=idx)


def test_well_specified_model_passes_all_coverage_tests(normal_series):
    """Historical VaR on iid normal data is correctly specified — it must pass.

    This is the positive control for the whole backtesting stack.
    """
    result = rolling_var_backtest(normal_series, confidence=0.95, window=500,
                                  method="parametric_normal")
    assert result.breach_rate == pytest.approx(0.05, abs=0.015)
    assert not any(t.reject_at_5pct for t in result.tests), result.summary()


def test_normal_var_fails_on_fat_tailed_clustered_data():
    """The headline result: a normal VaR is rejected on realistic returns.

    The synthetic generator produces Student-t shocks with volatility clustering.
    A normal-assumption model cannot cover that tail, and the coverage tests must
    detect it — otherwise the backtest is decorative.
    """
    prices = synthetic_prices(["A", "B", "C"], n_days=3000, seed=99, df=3.0)
    pnl = portfolio_returns(to_returns(prices))
    result = rolling_var_backtest(pnl, confidence=0.99, window=250,
                                  method="parametric_normal")
    assert result.breach_rate > 0.01
    assert any(t.reject_at_5pct for t in result.tests), result.summary()


def test_backtest_forecasts_use_only_past_data(normal_series):
    """No look-ahead: truncating the series must not change earlier forecasts.

    If a forecast at date t depended on data after t, cutting the tail off the
    input would change it.
    """
    full = rolling_var_backtest(normal_series, 0.99, 500, "historical")
    truncated = rolling_var_backtest(normal_series.iloc[:1500], 0.99, 500, "historical")
    overlap = truncated.var_forecasts.index
    pd.testing.assert_series_equal(
        full.var_forecasts.loc[overlap], truncated.var_forecasts, check_names=False
    )


def test_backtest_breach_flag_matches_the_forecast(normal_series):
    """A breach is exactly 'realised loss worse than forecast VaR'."""
    result = rolling_var_backtest(normal_series, 0.95, 500, "historical")
    expected = (-result.realised_returns > result.var_forecasts).astype(int)
    pd.testing.assert_series_equal(result.breaches, expected, check_names=False)


def test_higher_confidence_produces_fewer_breaches(normal_series):
    low = rolling_var_backtest(normal_series, 0.95, 500, "historical")
    high = rolling_var_backtest(normal_series, 0.99, 500, "historical")
    assert high.n_breaches < low.n_breaches
    assert (high.var_forecasts >= low.var_forecasts).all()


def test_student_t_model_covers_fat_tails_better_than_normal():
    """The t model should breach less often than the normal on fat-tailed data."""
    prices = synthetic_prices(["A", "B"], n_days=2500, seed=5, df=3.0)
    pnl = portfolio_returns(to_returns(prices))
    normal = rolling_var_backtest(pnl, 0.99, 250, "parametric_normal")
    student = rolling_var_backtest(pnl, 0.99, 250, "parametric_t")
    assert student.n_breaches <= normal.n_breaches


def test_backtest_rejects_insufficient_history(normal_series):
    with pytest.raises(ValueError, match="need more than"):
        rolling_var_backtest(normal_series.iloc[:100], 0.99, window=250)


def test_backtest_rejects_a_tiny_window(normal_series):
    with pytest.raises(ValueError, match="too short"):
        rolling_var_backtest(normal_series, 0.99, window=10)
