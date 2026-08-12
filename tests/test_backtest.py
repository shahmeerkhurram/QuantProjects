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
    _normal_tail,
    _null_shape,
    _standardised_t_tail,
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    basel_traffic_light,
    christoffersen_independence,
    compare_es_models,
    conditional_coverage,
    es_backtest,
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


# --------------------------------------------------------------------------
# Acerbi-Székely — Expected Shortfall backtesting
# --------------------------------------------------------------------------

def _flat_forecasts(n: int, sigma: float, alpha: float):
    """Constant VaR/ES forecasts from a normal of known scale."""
    var_unit, es_unit = _normal_tail(alpha)
    return np.full(n, sigma * var_unit), np.full(n, sigma * es_unit)


@pytest.mark.parametrize("statistic", ["z1", "z2"])
def test_es_statistic_is_centred_at_zero_under_a_correct_model(statistic):
    """The null itself: forecasts drawn from the same law that generates losses.

    Averaged over many independent samples the statistic must sit at zero. A
    non-zero centre would mean every downstream p-value is biased.
    """
    rng = np.random.default_rng(11)
    n, alpha, sigma = 4000, 0.025, 0.01
    var, es = _flat_forecasts(n, sigma, alpha)

    values = []
    for _ in range(150):
        losses = rng.standard_normal(n) * sigma
        values.append(
            acerbi_szekely_z1(losses, var, es)
            if statistic == "z1"
            else acerbi_szekely_z2(losses, var, es, 0.975)
        )
    assert np.nanmean(values) == pytest.approx(0.0, abs=0.02)


@pytest.mark.parametrize("statistic", ["z1", "z2"])
def test_es_statistic_is_negative_when_the_tail_is_understated(statistic):
    """Fat-tailed losses against normal forecasts: the failure ES exists to catch.

    Negative is the supervisory direction — realised tail losses exceed the
    forecast ES.
    """
    rng = np.random.default_rng(12)
    n, alpha, sigma = 4000, 0.025, 0.01
    var, es = _flat_forecasts(n, sigma, alpha)

    values = []
    for _ in range(100):
        # Unit-variance Student-t: same volatility, far heavier tail.
        losses = rng.standard_t(3, n) * np.sqrt(1 / 3) * sigma
        values.append(
            acerbi_szekely_z1(losses, var, es)
            if statistic == "z1"
            else acerbi_szekely_z2(losses, var, es, 0.975)
        )
    assert np.nanmean(values) < -0.02


def test_es_statistic_is_positive_when_the_model_is_over_conservative():
    """The other direction must be distinguishable, not merely 'not negative'."""
    rng = np.random.default_rng(13)
    n, alpha, sigma = 4000, 0.025, 0.01
    var, es = _flat_forecasts(n, sigma, alpha)
    losses = rng.standard_normal(n) * sigma * 0.5
    assert acerbi_szekely_z2(losses, var, es, 0.975) > 0.1


def test_z1_is_undefined_without_breaches():
    """A mean over an empty set is NaN, and is reported rather than hidden."""
    losses = np.zeros(100)
    var, es = _flat_forecasts(100, 0.01, 0.025)
    assert np.isnan(acerbi_szekely_z1(losses, var, es))


def test_null_shape_recovers_the_degrees_of_freedom_it_was_given():
    """The simulated null is only right if the tail shape is right.

    Matching the ES/VaR ratio must invert exactly, otherwise the critical values
    describe a different distribution from the one the model is claiming.
    """
    alpha = 0.025
    for df in (3.0, 4.0, 6.0, 10.0, 30.0):
        var, es = _standardised_t_tail(df, alpha)
        assert _null_shape(es / var, alpha) == pytest.approx(df, rel=1e-3)


def test_null_shape_falls_back_to_normal_for_a_thin_tail():
    alpha = 0.025
    var, es = _normal_tail(alpha)
    assert _null_shape(es / var, alpha) == float("inf")


def test_es_never_falls_below_var_in_any_forecast(normal_series):
    """ES is an average of losses beyond VaR, so it cannot be smaller.

    Asserted across every walk-forward forecast rather than at a single point,
    because a sign or indexing error would show up on only some days.
    """
    result = rolling_var_backtest(normal_series, 0.975, 500, "historical")
    assert result.es_forecasts is not None
    assert (result.es_forecasts >= result.var_forecasts).all()


def test_es_backtest_passes_a_correctly_specified_walk_forward_model():
    """Gaussian data, Gaussian model: the ES test must not reject."""
    rng = np.random.default_rng(14)
    series = pd.Series(
        rng.standard_normal(3000) * 0.01,
        index=pd.bdate_range("2010-01-01", periods=3000),
    )
    result = rolling_var_backtest(series, 0.975, 500, "parametric_normal")
    test = es_backtest(result, n_simulations=2000, seed=3)
    assert not test.reject_at_5pct
    assert test.p_value > 0.05
    assert test.n_simulations == 2000


def test_es_backtest_rejects_a_model_blind_to_the_tail():
    """Fat-tailed clustered data against a normal model: this must be caught."""
    prices = synthetic_prices(["A", "B", "C"], n_days=3000, seed=8, df=2.5)
    pnl = portfolio_returns(to_returns(prices))
    result = rolling_var_backtest(pnl, 0.975, 250, "parametric_normal")
    test = es_backtest(result, n_simulations=2000, seed=4)
    assert test.statistic < 0
    assert test.reject_at_5pct
    assert "understated" in test.interpretation


def test_es_backtest_requires_recorded_es_forecasts(normal_series):
    result = rolling_var_backtest(normal_series, 0.975, 500, "historical")
    result.es_forecasts = None
    with pytest.raises(ValueError, match="no ES forecasts"):
        es_backtest(result)


def test_es_backtest_rejects_an_unknown_test_name(normal_series):
    result = rolling_var_backtest(normal_series, 0.975, 500, "historical")
    with pytest.raises(ValueError, match="unknown test"):
        es_backtest(result, n_simulations=200, test="z3")


def test_es_backtest_rejects_too_few_simulations(normal_series):
    result = rolling_var_backtest(normal_series, 0.975, 500, "historical")
    with pytest.raises(ValueError, match="at least 100 simulations"):
        es_backtest(result, n_simulations=10)


def test_es_backtest_reports_an_untestable_model_rather_than_passing_it():
    """No breaches means no evidence, which is not the same as a clean bill.

    A model so conservative it is never breached would otherwise sail through an
    ES test by default — the one outcome a supervisor should not accept quietly.
    """
    index = pd.bdate_range("2010-01-01", periods=600)
    series = pd.Series(np.zeros(600), index=index)
    result = rolling_var_backtest(series, 0.975, 500, "historical")
    # A flat series produces zero-width forecasts; widen them so the ES test
    # sees positive forecasts that are never breached.
    result.var_forecasts = pd.Series(np.full(100, 0.05), index=index[500:])
    result.es_forecasts = pd.Series(np.full(100, 0.07), index=index[500:])
    result.realised_returns = pd.Series(np.zeros(100), index=index[500:])

    test = es_backtest(result, n_simulations=500, test="z1")
    assert np.isnan(test.statistic)
    assert not test.reject_at_5pct
    assert "untestable" in test.interpretation


def test_es_backtest_simulates_a_fat_tailed_null_when_the_model_claims_one():
    """The null must follow the model's own tail, not a normal by default.

    A Student-t model claims an ES/VaR ratio no normal can produce; simulating a
    normal null would reject it for being right. Comparing the critical value
    against a thin-tailed model's confirms the null actually moved.
    """
    prices = synthetic_prices(["A", "B"], n_days=2000, seed=21, df=3.0)
    pnl = portfolio_returns(to_returns(prices))
    fat = rolling_var_backtest(pnl, 0.975, 250, "parametric_t")
    thin = rolling_var_backtest(pnl, 0.975, 250, "parametric_normal")

    assert fat.es_forecasts is not None and thin.es_forecasts is not None
    fat_ratio = float((fat.es_forecasts / fat.var_forecasts).mean())
    thin_ratio = float((thin.es_forecasts / thin.var_forecasts).mean())
    assert fat_ratio > thin_ratio

    fat_test = es_backtest(fat, n_simulations=1000, seed=7)
    thin_test = es_backtest(thin, n_simulations=1000, seed=7)
    # A heavier null tail admits more extreme statistics before rejecting.
    assert fat_test.critical_value < thin_test.critical_value


def test_compare_es_models_tabulates_both_families_of_test():
    """The comparison table must carry the VaR verdict beside the ES verdict.

    That juxtaposition is the entire point: a model can pass one and fail the
    other, and a table showing only one of them would hide it.
    """
    prices = synthetic_prices(["A", "B", "C"], n_days=1200, seed=22, df=4.0)
    pnl = portfolio_returns(to_returns(prices))
    table = compare_es_models(
        pnl, confidence=0.975, window=250,
        methods=("historical", "parametric_normal"),
        n_simulations=500,
    )
    assert len(table) == 2
    assert set(table.columns) >= {
        "method", "z1", "z1_p", "z1_pass", "z2", "z2_p", "z2_pass",
        "kupiec_p", "christoffersen_p", "var_passes_all", "mean_es",
    }
    # ES must exceed VaR on average for every model, by construction.
    assert (table["mean_es"] > table["mean_var"]).all()
    assert table["z2_p"].is_monotonic_decreasing


def test_es_test_string_shows_the_statistic_and_verdict():
    prices = synthetic_prices(["A", "B"], n_days=1000, seed=23)
    pnl = portfolio_returns(to_returns(prices))
    result = rolling_var_backtest(pnl, 0.975, 250, "historical")
    rendered = str(es_backtest(result, n_simulations=500))
    assert "Acerbi-Székely" in rendered
    assert ("pass" in rendered) or ("REJECT" in rendered)
