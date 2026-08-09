"""Value-at-Risk and Expected Shortfall estimators.

Sign convention
---------------
VaR and ES are reported as **positive loss magnitudes**. A 1-day 99% VaR of
``0.0312`` means "on 1 day in 100 we expect to lose at least 3.12% of portfolio
value". This is the convention risk reports use, and it avoids the constant
sign confusion of quoting a negative return quantile.

Why three Monte Carlo engines
-----------------------------
A Monte Carlo VaR that samples from ``N(mu_p, sigma_p)`` fitted to the *portfolio*
series is not an independent method: it converges to the parametric-normal
answer by construction and adds nothing but simulation noise. The engines here
are all genuinely different from the parametric number because each relaxes a
different assumption:

``gaussian``   draws the *asset* vector from a multivariate normal via a Cholesky
               factor of the covariance matrix, then re-aggregates through the
               portfolio weights. This is the one case that does agree with
               parametric normal in the limit — it is kept as a control that
               proves the simulation machinery is unbiased.
``student_t``  same correlation structure but fat marginal tails, with the
               degrees of freedom fitted from the data. Produces a materially
               larger tail loss than the normal.
``bootstrap``  resamples historical *dates* whole, so the empirical cross-asset
               dependence and tail co-movement survive without any distributional
               assumption at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "RiskResult",
    "conditional_var",
    "historical_var",
    "monte_carlo_var",
    "parametric_var",
    "portfolio_returns",
    "scale_horizon",
]

MonteCarloEngine = Literal["gaussian", "student_t", "bootstrap"]


@dataclass(frozen=True)
class RiskResult:
    """A single VaR/ES estimate with the assumptions that produced it."""

    method: str
    confidence: float
    var: float
    expected_shortfall: float
    horizon_days: int = 1
    observations: int | None = None
    detail: str = ""

    def as_row(self) -> dict[str, object]:
        return {
            "method": self.method,
            "confidence": self.confidence,
            "horizon_days": self.horizon_days,
            "VaR": self.var,
            "ES": self.expected_shortfall,
            "observations": self.observations,
            "detail": self.detail,
        }


def _check_confidence(confidence: float) -> None:
    if not 0.5 <= confidence < 1.0:
        raise ValueError(f"confidence must lie in [0.5, 1.0), got {confidence}")


def _as_array(returns) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("return series is empty")
    if not np.isfinite(arr).all():
        raise ValueError("return series contains NaN or inf; clean the data first")
    return arr


def portfolio_returns(asset_returns: pd.DataFrame, weights=None) -> pd.Series:
    """Aggregate asset returns into a portfolio series.

    Weights default to equal-weight and are normalised to sum to one. This is a
    constant-weight (daily-rebalanced) portfolio, which is the standard
    assumption for a risk report — it is not a buy-and-hold backtest.
    """
    if asset_returns.empty:
        raise ValueError("asset_returns is empty")
    n = asset_returns.shape[1]
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != n:
            raise ValueError(f"expected {n} weights for {n} assets, got {w.size}")
        total = w.sum()
        if np.isclose(total, 0.0):
            raise ValueError("weights sum to zero")
        w = w / total
    return pd.Series(asset_returns.to_numpy() @ w, index=asset_returns.index, name="portfolio")


def _es_from_losses(losses: np.ndarray, var: float) -> float:
    """Mean loss conditional on breaching VaR.

    Falls back to the VaR itself when the tail sample is empty, which can happen
    for tiny samples at extreme confidence levels.
    """
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else float(var)


def historical_var(returns, confidence: float = 0.99, horizon_days: int = 1) -> RiskResult:
    """Empirical quantile of the loss distribution. No distributional assumption.

    Uses the ``lower`` interpolation so the reported VaR is always an observed
    loss rather than an interpolated value between two observations — the
    conservative choice, and it keeps the breach count consistent with the
    backtest.
    """
    _check_confidence(confidence)
    losses = -_as_array(returns)
    var = float(np.quantile(losses, confidence, method="lower"))
    result = RiskResult(
        method="historical",
        confidence=confidence,
        var=var,
        expected_shortfall=_es_from_losses(losses, var),
        observations=losses.size,
        detail="empirical quantile",
    )
    return scale_horizon(result, horizon_days)


def parametric_var(
    returns,
    confidence: float = 0.99,
    horizon_days: int = 1,
    distribution: Literal["normal", "student_t"] = "normal",
) -> RiskResult:
    """Closed-form VaR/ES under a fitted normal or Student-t.

    The Student-t variant matters in practice: daily equity returns are
    leptokurtic, so the normal assumption systematically understates the tail,
    which is exactly what the Kupiec test in :mod:`risk_engine.backtest` detects.
    """
    _check_confidence(confidence)
    arr = _as_array(returns)
    mu, sigma = float(arr.mean()), float(arr.std(ddof=1))
    alpha = 1.0 - confidence

    if distribution == "normal":
        z = stats.norm.ppf(confidence)
        var = -(mu - sigma * z)
        # ES of a normal: mu - sigma * phi(z)/alpha, negated into a loss.
        es = -(mu - sigma * stats.norm.pdf(z) / alpha)
        detail = f"normal(mu={mu:.6f}, sigma={sigma:.6f})"
    elif distribution == "student_t":
        df, loc, scale = stats.t.fit(arr)
        if df <= 1:
            raise ValueError(f"fitted t has df={df:.2f}; mean and ES are undefined")
        t_q = stats.t.ppf(alpha, df)
        var = -(loc + scale * t_q)
        # Standard closed form for the t expected shortfall.
        es_std = (df + t_q**2) / (df - 1) * stats.t.pdf(t_q, df) / alpha
        es = -(loc - scale * es_std)
        detail = f"student_t(df={df:.2f}, loc={loc:.6f}, scale={scale:.6f})"
    else:
        raise ValueError(f"unknown distribution {distribution!r}")

    result = RiskResult(
        method=f"parametric_{distribution}",
        confidence=confidence,
        var=float(var),
        expected_shortfall=float(es),
        observations=arr.size,
        detail=detail,
    )
    return scale_horizon(result, horizon_days)


def monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights=None,
    confidence: float = 0.99,
    horizon_days: int = 1,
    engine: MonteCarloEngine = "student_t",
    n_paths: int = 100_000,
    seed: int | None = 42,
) -> RiskResult:
    """Simulate the *asset* return vector, then aggregate through the weights.

    Simulating assets rather than the pre-aggregated portfolio series is what
    makes this an independent estimate: the correlation structure is modelled
    explicitly, so the result responds to diversification and to tail dependence
    instead of merely resampling a fitted portfolio sigma.

    See the module docstring for what each ``engine`` assumes.
    """
    _check_confidence(confidence)
    if asset_returns.empty:
        raise ValueError("asset_returns is empty")
    if n_paths < 1000:
        raise ValueError(f"n_paths={n_paths} is too small for a stable tail quantile")

    rng = np.random.default_rng(seed)
    data = asset_returns.to_numpy(dtype=float)
    n_assets = data.shape[1]

    if weights is None:
        w = np.full(n_assets, 1.0 / n_assets)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != n_assets:
            raise ValueError(f"expected {n_assets} weights, got {w.size}")
        w = w / w.sum()

    if engine == "bootstrap":
        # Resample whole dates so cross-asset co-movement is preserved exactly.
        idx = rng.integers(0, data.shape[0], size=n_paths)
        simulated = data[idx] @ w
        detail = "iid bootstrap of historical dates"
    else:
        mu = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)
        cov = np.atleast_2d(cov)
        chol = _safe_cholesky(cov)

        if engine == "gaussian":
            shocks = rng.standard_normal((n_paths, n_assets)) @ chol.T
            detail = "multivariate normal via Cholesky"
        elif engine == "student_t":
            # Fit one df per asset and take the most conservative (fattest) tail,
            # so a single df drives the shared t-copula chi-square mixture.
            dfs = [float(stats.t.fit(data[:, i])[0]) for i in range(n_assets)]
            df = max(min(dfs), 2.1)  # variance requires df > 2
            normal = rng.standard_normal((n_paths, n_assets)) @ chol.T
            chi = rng.chisquare(df, size=(n_paths, 1)) / df
            # Rescale so the simulated covariance still matches the sample.
            shocks = normal / np.sqrt(chi) * np.sqrt((df - 2.0) / df)
            detail = f"multivariate student-t (df={df:.2f}) via Cholesky"
        else:
            raise ValueError(f"unknown engine {engine!r}")

        simulated = (mu + shocks) @ w

    losses = -simulated
    var = float(np.quantile(losses, confidence))
    result = RiskResult(
        method=f"monte_carlo_{engine}",
        confidence=confidence,
        var=var,
        expected_shortfall=_es_from_losses(losses, var),
        observations=n_paths,
        detail=detail,
    )
    return scale_horizon(result, horizon_days)


def conditional_var(
    returns,
    confidence: float = 0.99,
    horizon_days: int = 1,
    model: Literal["ewma", "garch"] = "ewma",
    innovation: Literal["empirical", "normal", "student_t"] = "empirical",
    lam: float = 0.94,
    garch_params=None,
) -> RiskResult:
    """VaR/ES from a conditional volatility forecast.

    The estimate is ``sigma_{t+1} * q_alpha(z)``, where ``sigma_{t+1}`` comes
    from an EWMA or GARCH(1,1) filter and ``q_alpha(z)`` is the tail quantile of
    the standardised residuals.

    With ``innovation="empirical"`` this is **Filtered Historical Simulation** —
    the volatility model handles clustering, the empirical residual quantile
    handles fat tails, and neither has to approximate the other. That separation
    is why FHS outperforms both the unconditional models and a
    conditional-normal one.

    Parameters
    ----------
    garch_params
        A pre-fitted :class:`~risk_engine.volatility.GarchParams`. Supplying one
        skips the maximum-likelihood fit, which matters in a walk-forward
        backtest where refitting on every step would dominate the runtime.
    """
    from .volatility import (
        ewma_forecast,
        ewma_variance,
        fit_garch11,
        garch_forecast,
        garch_variance,
        standardised_residuals,
    )

    _check_confidence(confidence)
    arr = _as_array(returns)
    if arr.size < 30:
        raise ValueError(
            f"need at least 30 observations to filter volatility, got {arr.size}"
        )

    if model == "ewma":
        variance = ewma_variance(arr, lam)
        sigma_next = ewma_forecast(arr, lam)
        detail = f"EWMA(lambda={lam})"
    elif model == "garch":
        params = garch_params if garch_params is not None else fit_garch11(arr)
        variance = garch_variance(arr, params)
        sigma_next = garch_forecast(arr, params)
        detail = f"GARCH(1,1) persistence={params.persistence:.4f}"
    else:
        raise ValueError(f"unknown volatility model {model!r}")

    z = standardised_residuals(arr, variance)

    if innovation == "empirical":
        z_losses = -z
        q_var = float(np.quantile(z_losses, confidence, method="lower"))
        tail = z_losses[z_losses >= q_var]
        q_es = float(tail.mean()) if tail.size else q_var
        detail += " + empirical residual quantile (FHS)"
    elif innovation == "normal":
        z_score = stats.norm.ppf(confidence)
        q_var = z_score
        q_es = stats.norm.pdf(z_score) / (1.0 - confidence)
        detail += " + normal innovations"
    elif innovation == "student_t":
        df, loc, scale = stats.t.fit(z)
        if df <= 1:
            raise ValueError(f"fitted residual t has df={df:.2f}; ES is undefined")
        alpha = 1.0 - confidence
        t_q = stats.t.ppf(alpha, df)
        q_var = -(loc + scale * t_q)
        q_es = -(loc - scale * ((df + t_q**2) / (df - 1) * stats.t.pdf(t_q, df) / alpha))
        detail += f" + student-t innovations (df={df:.2f})"
    else:
        raise ValueError(f"unknown innovation distribution {innovation!r}")

    result = RiskResult(
        method=f"conditional_{model}_{innovation}",
        confidence=confidence,
        var=float(sigma_next * q_var),
        expected_shortfall=float(sigma_next * q_es),
        observations=arr.size,
        detail=f"{detail}; sigma_next={sigma_next:.6f}",
    )
    return scale_horizon(result, horizon_days)


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Cholesky factor, repairing a covariance matrix that is only near-PSD.

    Sample covariance matrices from short windows can pick up tiny negative
    eigenvalues from floating-point error. Clipping the spectrum is preferable
    to failing the whole risk run.
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        repaired = eigvecs @ np.diag(np.clip(eigvals, 1e-12, None)) @ eigvecs.T
        return np.linalg.cholesky(repaired)


def scale_horizon(result: RiskResult, horizon_days: int) -> RiskResult:
    """Apply the square-root-of-time rule to a 1-day estimate.

    Valid only for iid returns with no autocorrelation and no drift. It is the
    Basel convention and it is what regulators expect to see, but it understates
    risk under volatility clustering — stated here because the assumption is
    load-bearing and frequently left implicit.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    if horizon_days == 1:
        return result
    factor = np.sqrt(horizon_days)
    return RiskResult(
        method=result.method,
        confidence=result.confidence,
        var=result.var * factor,
        expected_shortfall=result.expected_shortfall * factor,
        horizon_days=horizon_days,
        observations=result.observations,
        detail=f"{result.detail}; sqrt-time scaled from 1d",
    )
