"""VaR and Expected Shortfall tests.

The strategy is *known-truth testing*: generate data from a distribution whose
tail quantile is known analytically, then assert the estimator recovers it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_engine.var import (
    historical_var,
    monte_carlo_var,
    parametric_var,
    portfolio_returns,
    scale_horizon,
)


@pytest.fixture
def normal_returns() -> pd.Series:
    """40,000 draws from N(0, 1%) — enough for a tight tail estimate."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0, 0.01, 40_000))


@pytest.fixture
def normal_assets() -> pd.DataFrame:
    """Three correlated assets, jointly normal, with a known covariance."""
    rng = np.random.default_rng(1)
    corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    chol = np.linalg.cholesky(corr)
    raw = rng.standard_normal((30_000, 3)) @ chol.T
    return pd.DataFrame(raw * 0.012, columns=["A", "B", "C"])


def test_parametric_normal_recovers_analytic_quantile(normal_returns):
    """For N(0, sigma) the 99% VaR is exactly z_{0.99} * sigma."""
    expected = stats.norm.ppf(0.99) * 0.01
    assert parametric_var(normal_returns, 0.99).var == pytest.approx(expected, rel=0.02)


def test_historical_matches_parametric_on_normal_data(normal_returns):
    """With a large normal sample the two methods must agree.

    Disagreement here would mean one of them has a sign or quantile bug.
    """
    hist = historical_var(normal_returns, 0.99).var
    para = parametric_var(normal_returns, 0.99).var
    assert hist == pytest.approx(para, rel=0.03)


def test_expected_shortfall_always_exceeds_var(normal_returns):
    """ES is the mean loss *beyond* VaR, so it is strictly larger by definition."""
    for confidence in (0.90, 0.95, 0.99):
        for result in (
            historical_var(normal_returns, confidence),
            parametric_var(normal_returns, confidence, distribution="normal"),
            parametric_var(normal_returns, confidence, distribution="student_t"),
        ):
            assert result.expected_shortfall > result.var, result.method


def test_normal_es_matches_closed_form(normal_returns):
    """ES of a normal is sigma * phi(z) / alpha."""
    alpha = 0.01
    z = stats.norm.ppf(0.99)
    expected = 0.01 * stats.norm.pdf(z) / alpha
    assert parametric_var(normal_returns, 0.99).expected_shortfall == pytest.approx(
        expected, rel=0.03
    )


def test_var_increases_with_confidence(normal_returns):
    levels = [0.90, 0.95, 0.975, 0.99]
    values = [historical_var(normal_returns, c).var for c in levels]
    assert all(b > a for a, b in itertools.pairwise(values))


def test_student_t_reports_fatter_tail_than_normal():
    """On leptokurtic data the t-fit must produce a larger 99% VaR.

    This is the whole reason the t variant exists: a normal fit to fat-tailed
    returns systematically understates the tail.
    """
    rng = np.random.default_rng(3)
    fat = pd.Series(stats.t.rvs(df=3, size=40_000, random_state=rng) * 0.006)
    normal_var = parametric_var(fat, 0.99, distribution="normal").var
    t_var = parametric_var(fat, 0.99, distribution="student_t").var
    assert t_var > normal_var * 1.05


def test_monte_carlo_gaussian_agrees_with_parametric(normal_assets):
    """The gaussian engine is a control: it must reproduce the parametric answer.

    If it did not, the Cholesky/aggregation machinery would be biased, and the
    t and bootstrap engines built on it could not be trusted either.
    """
    pnl = portfolio_returns(normal_assets)
    parametric = parametric_var(pnl, 0.99).var
    simulated = monte_carlo_var(normal_assets, confidence=0.99, engine="gaussian",
                                n_paths=200_000, seed=5).var
    assert simulated == pytest.approx(parametric, rel=0.03)


def test_monte_carlo_bootstrap_agrees_with_historical(normal_assets):
    """Resampling historical dates must reproduce the historical quantile."""
    pnl = portfolio_returns(normal_assets)
    hist = historical_var(pnl, 0.99).var
    boot = monte_carlo_var(normal_assets, confidence=0.99, engine="bootstrap",
                           n_paths=200_000, seed=6).var
    assert boot == pytest.approx(hist, rel=0.05)


def test_monte_carlo_t_engine_exceeds_gaussian_on_fat_tails():
    """On fat-tailed assets the t engine must find a bigger loss than the normal.

    This is the specific failure the original notebook had: a Monte Carlo that
    drew from a fitted normal could never disagree with the parametric number.
    """
    rng = np.random.default_rng(11)
    data = pd.DataFrame(
        stats.t.rvs(df=3, size=(20_000, 3), random_state=rng) * 0.008,
        columns=["A", "B", "C"],
    )
    gaussian = monte_carlo_var(data, confidence=0.99, engine="gaussian",
                               n_paths=100_000, seed=7).var
    student = monte_carlo_var(data, confidence=0.99, engine="student_t",
                              n_paths=100_000, seed=7).var
    assert student > gaussian


def test_monte_carlo_respects_diversification():
    """Two uncorrelated assets must carry less risk than two perfectly correlated ones.

    A portfolio-level simulation cannot express this; an asset-level one must.
    """
    rng = np.random.default_rng(13)
    base = rng.standard_normal(20_000) * 0.01
    independent = pd.DataFrame({"A": base, "B": rng.standard_normal(20_000) * 0.01})
    identical = pd.DataFrame({"A": base, "B": base})

    diversified = monte_carlo_var(independent, confidence=0.99, engine="gaussian",
                                  n_paths=100_000, seed=9).var
    concentrated = monte_carlo_var(identical, confidence=0.99, engine="gaussian",
                                   n_paths=100_000, seed=9).var
    assert diversified < concentrated * 0.85


def test_horizon_scaling_uses_square_root_of_time(normal_returns):
    one_day = historical_var(normal_returns, 0.99, horizon_days=1)
    ten_day = historical_var(normal_returns, 0.99, horizon_days=10)
    assert ten_day.var == pytest.approx(one_day.var * np.sqrt(10))
    assert ten_day.horizon_days == 10


def test_portfolio_returns_normalise_weights():
    frame = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]})
    # Weights 2:2 are equal after normalisation, so this equals the simple mean.
    assert portfolio_returns(frame, [2.0, 2.0]).tolist() == pytest.approx([0.02, 0.03])


def test_portfolio_returns_handles_negative_weights():
    """A long/short book is legitimate; only a zero-sum weight vector is not."""
    frame = pd.DataFrame({"A": [0.10], "B": [0.02]})
    assert portfolio_returns(frame, [2.0, -1.0]).iloc[0] == pytest.approx(0.18)
    with pytest.raises(ValueError, match="sum to zero"):
        portfolio_returns(frame, [1.0, -1.0])


@pytest.mark.parametrize("confidence", [0.0, 0.4, 1.0, 1.5])
def test_invalid_confidence_rejected(normal_returns, confidence):
    with pytest.raises(ValueError, match="confidence"):
        historical_var(normal_returns, confidence)


def test_nan_input_rejected():
    with pytest.raises(ValueError, match="NaN"):
        historical_var(pd.Series([0.01, np.nan, -0.02]), 0.95)


def test_scale_horizon_rejects_zero():
    with pytest.raises(ValueError, match="horizon_days"):
        scale_horizon(historical_var(pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))), 0)
