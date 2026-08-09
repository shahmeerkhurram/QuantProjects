"""Conditional volatility and Filtered Historical Simulation tests.

The decisive test in this file is
:func:`test_volatility_filter_fixes_breach_clustering` — it asserts the thing
the whole module exists to do. Everything else supports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.backtest import (
    basel_zone_thresholds,
    christoffersen_independence,
    compare_models,
    rolling_var_backtest,
)
from risk_engine.var import conditional_var, historical_var, parametric_var
from risk_engine.volatility import (
    ewma_forecast,
    ewma_variance,
    fit_garch11,
    garch_forecast,
    garch_variance,
    standardised_residuals,
)


def simulate_garch(n=4000, omega=1e-6, alpha=0.09, beta=0.90, seed=1, df=None):
    """Generate a series from a known GARCH(1,1) process.

    Known parameters mean the fit has a correct answer to recover, and the
    volatility clustering is real rather than assumed.
    """
    rng = np.random.default_rng(seed)
    var = omega / (1 - alpha - beta)
    out = np.empty(n)
    for t in range(n):
        # Unit-variance innovations either way, so alpha/beta keep their meaning.
        shock = (
            rng.standard_normal()
            if df is None
            else rng.standard_t(df) * np.sqrt((df - 2) / df)
        )
        out[t] = np.sqrt(var) * shock
        var = omega + alpha * out[t] ** 2 + beta * var
    return out


# --------------------------------------------------------------------------
# EWMA
# --------------------------------------------------------------------------

def test_ewma_matches_the_recursion_by_hand():
    """v_t = lam*v_{t-1} + (1-lam)*r_{t-1}^2, seeded with the sample variance."""
    r = np.array([0.01, -0.02, 0.015, -0.005, 0.03])
    lam = 0.94
    got = ewma_variance(r, lam)

    expected = np.empty(5)
    expected[0] = np.var(r, ddof=1)
    for t in range(1, 5):
        expected[t] = lam * expected[t - 1] + (1 - lam) * r[t - 1] ** 2
    assert got == pytest.approx(expected)


def test_ewma_uses_no_look_ahead():
    """Element t must not depend on r_t — truncating the series proves it."""
    r = simulate_garch(400)
    full = ewma_variance(r)
    truncated = ewma_variance(r[:200])
    assert full[:200] == pytest.approx(truncated)


def test_ewma_volatility_reacts_to_a_shock():
    """A large return must raise the next forecast; that is the entire point."""
    calm = np.full(300, 0.001)
    shocked = np.concatenate([calm, [0.15]])
    assert ewma_forecast(shocked) > ewma_forecast(calm) * 5


def test_ewma_forecast_decays_back_after_a_shock():
    """With no further news, volatility must revert toward the calm level."""
    calm = np.full(300, 0.001)
    right_after = ewma_forecast(np.concatenate([calm, [0.15]]))
    later = ewma_forecast(np.concatenate([calm, [0.15], np.full(60, 0.001)]))
    assert later < right_after


@pytest.mark.parametrize("lam", [0.0, 1.0, -0.5, 1.2])
def test_invalid_lambda_rejected(lam):
    with pytest.raises(ValueError, match="lambda"):
        ewma_variance(np.random.default_rng(0).normal(0, 0.01, 100), lam)


# --------------------------------------------------------------------------
# GARCH(1,1)
# --------------------------------------------------------------------------

def test_garch_fit_recovers_known_parameters():
    """The strongest possible check: simulate from known parameters, refit.

    Tolerances are loose because 6,000 observations still leave real sampling
    error in a GARCH MLE — but they are tight enough that a wrong likelihood or
    a broken recursion would fail.
    """
    r = simulate_garch(n=6000, omega=1e-6, alpha=0.09, beta=0.90, seed=1)
    fit = fit_garch11(r)
    assert fit.converged
    assert fit.alpha == pytest.approx(0.09, abs=0.03)
    assert fit.beta == pytest.approx(0.90, abs=0.04)
    assert fit.omega == pytest.approx(1e-6, rel=0.6)


def test_fitted_process_is_stationary():
    """alpha + beta < 1, otherwise the long-run variance does not exist."""
    fit = fit_garch11(simulate_garch(3000, seed=4))
    assert 0.0 < fit.persistence < 1.0
    assert np.isfinite(fit.long_run_variance)
    assert fit.long_run_variance > 0


def test_high_persistence_means_a_long_half_life():
    """Half-life is the interpretable summary of how slowly shocks decay."""
    fit = fit_garch11(simulate_garch(4000, alpha=0.05, beta=0.94, seed=8))
    assert fit.persistence > 0.95
    assert fit.half_life > 10


def test_garch_likelihood_beats_a_constant_variance_model():
    """On genuinely heteroskedastic data GARCH must fit better than homoskedastic.

    Compares against the Gaussian log-likelihood of a constant-variance model,
    which is the nested special case alpha = beta = 0.
    """
    r = simulate_garch(3000, seed=12)
    fit = fit_garch11(r)

    var = np.var(r, ddof=1)
    constant_ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(var) + r**2 / var)
    assert fit.log_likelihood > constant_ll


def test_garch_variance_has_no_look_ahead():
    r = simulate_garch(500, seed=3)
    fit = fit_garch11(r)
    full = garch_variance(r, fit)
    truncated = garch_variance(r[:250], fit)
    assert full[:250] == pytest.approx(truncated)


def test_garch_forecast_is_positive_and_finite():
    r = simulate_garch(1000, seed=6)
    assert 0 < garch_forecast(r, fit_garch11(r)) < 1.0


# --------------------------------------------------------------------------
# Standardised residuals
# --------------------------------------------------------------------------

def test_standardised_residuals_have_roughly_unit_variance():
    """If the filter is working, dividing by sigma_t normalises the scale."""
    r = simulate_garch(4000, seed=7)
    z = standardised_residuals(r, garch_variance(r, fit_garch11(r)))
    assert np.std(z, ddof=1) == pytest.approx(1.0, abs=0.12)


def test_filtering_removes_volatility_clustering():
    """Squared returns are autocorrelated; squared residuals should not be.

    Autocorrelation of squared values is the standard diagnostic for
    heteroskedasticity, so this measures the filter's actual job.
    """
    r = simulate_garch(4000, seed=9)
    z = standardised_residuals(r, garch_variance(r, fit_garch11(r)))

    def lag1_autocorr(x):
        x = x - x.mean()
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    raw = lag1_autocorr(r**2)
    filtered = lag1_autocorr(z**2)
    assert raw > 0.05
    assert abs(filtered) < raw / 2


def test_mismatched_variance_shape_rejected():
    with pytest.raises(ValueError, match="shape"):
        standardised_residuals(np.ones(10) * 0.01, np.ones(5))


# --------------------------------------------------------------------------
# Conditional VaR
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["ewma", "garch"])
@pytest.mark.parametrize("innovation", ["empirical", "normal", "student_t"])
def test_conditional_var_is_positive_with_es_above_it(model, innovation):
    r = pd.Series(simulate_garch(1200, seed=10))
    result = conditional_var(r, 0.99, model=model, innovation=innovation)
    assert result.var > 0
    assert result.expected_shortfall > result.var


def test_conditional_var_responds_to_recent_volatility():
    """The defining property: identical history, different recent conditions.

    An unconditional model gives nearly the same answer for both series because
    it averages over the whole window. A conditional one must not.
    """
    rng = np.random.default_rng(11)
    base = rng.normal(0, 0.01, 500)
    calm_end = np.concatenate([base, rng.normal(0, 0.002, 60)])
    wild_end = np.concatenate([base, rng.normal(0, 0.05, 60)])

    calm = conditional_var(calm_end, 0.99, model="ewma").var
    wild = conditional_var(wild_end, 0.99, model="ewma").var
    assert wild > calm * 3

    # The unconditional model barely distinguishes them.
    unconditional_ratio = (
        historical_var(wild_end, 0.99).var / historical_var(calm_end, 0.99).var
    )
    assert unconditional_ratio < wild / calm


def test_fhs_finds_a_fatter_tail_than_conditional_normal():
    """With fat-tailed innovations, the empirical residual quantile must be larger."""
    r = simulate_garch(3000, seed=13, df=4)
    normal = conditional_var(r, 0.99, model="ewma", innovation="normal").var
    fhs = conditional_var(r, 0.99, model="ewma", innovation="empirical").var
    assert fhs > normal


def test_conditional_var_needs_enough_history():
    with pytest.raises(ValueError, match="at least 30"):
        conditional_var(np.random.default_rng(0).normal(0, 0.01, 20), 0.99)


@pytest.mark.parametrize("bad", ["arch", "stochvol"])
def test_unknown_volatility_model_rejected(bad):
    with pytest.raises(ValueError, match="unknown volatility model"):
        conditional_var(simulate_garch(200), 0.99, model=bad)


# --------------------------------------------------------------------------
# Basel zones — now derived rather than hardcoded
# --------------------------------------------------------------------------

def test_basel_thresholds_reproduce_the_supervisory_table():
    """250 days at 99% must give the published green 0-4 / amber 5-9 zones."""
    assert basel_zone_thresholds(250, 0.99) == (4, 9)


def test_basel_thresholds_scale_with_confidence():
    """At 95% you expect 12.5 breaches per 250 days, so the zones must widen.

    The previous implementation ignored `confidence` entirely and would have
    called 10 breaches at 95% a red-zone failure, when it is comfortably normal.
    """
    green_99, _ = basel_zone_thresholds(250, 0.99)
    green_95, _ = basel_zone_thresholds(250, 0.95)
    assert green_95 > green_99
    assert green_95 >= 12


def test_basel_thresholds_grow_with_sample_length():
    green_250, _ = basel_zone_thresholds(250, 0.99)
    green_1000, _ = basel_zone_thresholds(1000, 0.99)
    assert green_1000 > green_250
    # Sub-linear: the binomial tail does not scale proportionally with n.
    assert green_1000 < green_250 * 4


# --------------------------------------------------------------------------
# The headline claim
# --------------------------------------------------------------------------

def test_volatility_filter_fixes_breach_clustering():
    """The reason this module exists.

    On data with genuine volatility clustering, an unconditional model's
    breaches cluster and Christoffersen rejects. A volatility-filtered model
    must fix that — the breaches should become independent.
    """
    r = pd.Series(simulate_garch(3000, seed=21, df=5),
                  index=pd.bdate_range("2013-01-01", periods=3000))

    unconditional = rolling_var_backtest(r, 0.99, 250, "parametric_normal")
    filtered = rolling_var_backtest(r, 0.99, 250, "ewma_fhs")

    uncond_ind = christoffersen_independence(unconditional.breaches.to_numpy())
    filt_ind = christoffersen_independence(filtered.breaches.to_numpy())

    assert filt_ind.p_value > uncond_ind.p_value, (
        f"filtering should reduce clustering: unconditional p={uncond_ind.p_value:.4f}, "
        f"filtered p={filt_ind.p_value:.4f}"
    )
    assert not filt_ind.reject_at_5pct


def test_filtered_model_tracks_changing_volatility():
    """A conditional VaR series must actually vary; an unconditional one crawls.

    Measured by coefficient of variation of the forecast series.
    """
    r = pd.Series(simulate_garch(2000, seed=23),
                  index=pd.bdate_range("2015-01-01", periods=2000))
    uncond = rolling_var_backtest(r, 0.99, 250, "parametric_normal").var_forecasts
    filtered = rolling_var_backtest(r, 0.99, 250, "ewma_fhs").var_forecasts

    assert filtered.std() / filtered.mean() > uncond.std() / uncond.mean()


def test_compare_models_returns_one_row_per_method():
    r = pd.Series(simulate_garch(1200, seed=25),
                  index=pd.bdate_range("2018-01-01", periods=1200))
    methods = ("historical", "parametric_normal", "ewma_fhs")
    table = compare_models(r, 0.99, 250, methods)

    assert list(table["method"].sort_values()) == sorted(methods)
    assert table["joint_p"].is_monotonic_decreasing  # sorted best-first
    assert set(table.columns) >= {"breaches", "breach_rate", "kupiec_p", "passes_all"}


def test_parametric_var_unchanged_by_the_new_code():
    """Regression guard: adding conditional models must not alter existing ones."""
    rng = np.random.default_rng(31)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    from scipy import stats as sps

    assert parametric_var(r, 0.99).var == pytest.approx(
        sps.norm.ppf(0.99) * 0.01, rel=0.03
    )
