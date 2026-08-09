"""Conditional volatility models — EWMA and GARCH(1,1).

Why this module exists
----------------------
The unconditional VaR models in :mod:`risk_engine.var` all fail their coverage
tests on real equity data, and — importantly — they fail the *independence* test
as well as the coverage one. Breaches arrive in clusters. That is the signature
of a specific, diagnosable defect: the models assume volatility is constant when
it plainly is not.

The fix is to forecast tomorrow's volatility rather than averaging over the
whole window, then scale the tail quantile by it:

    VaR_{t+1} = -sigma_{t+1} * q_alpha(z)

where ``z`` are the standardised residuals ``r_t / sigma_t``. Taking ``q_alpha``
from the *empirical* distribution of ``z`` rather than from a normal is known as
**Filtered Historical Simulation** (Barone-Adesi et al., 1999). It is the
industry-standard approach because it separates the two problems cleanly:
the GARCH/EWMA filter handles volatility clustering, and the empirical quantile
of the residuals handles fat tails, with neither having to model the other.

Two filters are provided:

``EWMA``   RiskMetrics' exponentially weighted moving average. One fixed
           parameter (``lambda = 0.94`` for daily data), nothing to estimate, so
           it cannot overfit and is cheap enough to refit on every step of a
           walk-forward backtest.
``GARCH``  GARCH(1,1) fitted by maximum likelihood. Strictly more general — it
           has a mean-reverting long-run variance that EWMA lacks — at the cost
           of an optimisation on every refit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

__all__ = [
    "GarchParams",
    "ewma_forecast",
    "ewma_variance",
    "fit_garch11",
    "garch_forecast",
    "garch_variance",
    "standardised_residuals",
]

# RiskMetrics' decay factor for daily data. Not fitted — that is the point of it.
RISKMETRICS_LAMBDA = 0.94


def _clean(returns) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"need at least 2 observations, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError("return series contains NaN or inf")
    return arr


#: Observations used to seed the EWMA recursion before it becomes self-sustaining.
DEFAULT_BURN_IN = 30


def ewma_variance(
    returns, lam: float = RISKMETRICS_LAMBDA, burn_in: int = DEFAULT_BURN_IN
) -> np.ndarray:
    """Conditional variance path under an EWMA filter.

    The recursion is ``v_t = lam * v_{t-1} + (1 - lam) * r_{t-1}^2``, so element
    ``t`` uses only returns strictly before ``t``.

    **Seeding.** The recursion needs a starting variance. Using the variance of
    the *whole* series would leak future information into every element — a
    genuine look-ahead bug, and a subtle one, because the resulting series still
    looks perfectly plausible. Instead the seed is the variance of the first
    ``burn_in`` observations only. Consequently:

    * elements ``0 .. burn_in - 1`` are warm-up, contaminated by the seed;
    * elements ``burn_in`` onward are strictly causal.

    The seed's influence decays as ``lam ** t`` — at ``lam = 0.94`` it is below
    1e-7 after 250 steps, so for a standard risk window the warm-up is
    immaterial. It is documented rather than hidden because "immaterial" is a
    judgement the reader should be able to check.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lambda must lie in (0, 1), got {lam}")
    r = _clean(returns)
    if burn_in < 2:
        raise ValueError(f"burn_in must be at least 2, got {burn_in}")

    seed_block = r[: min(burn_in, r.size)]
    seed = float(np.var(seed_block, ddof=1))
    if seed <= 0:
        # A constant opening block would give a zero seed and divide-by-zero
        # residuals downstream; fall back to the mean square of the block.
        seed = float(np.mean(seed_block**2)) or 1e-12

    var = np.empty(r.size)
    var[0] = seed
    for t in range(1, r.size):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return var


def ewma_forecast(
    returns, lam: float = RISKMETRICS_LAMBDA, burn_in: int = DEFAULT_BURN_IN
) -> float:
    """One-step-ahead volatility forecast for the period after the sample ends."""
    r = _clean(returns)
    var = ewma_variance(r, lam, burn_in)
    next_var = lam * var[-1] + (1.0 - lam) * r[-1] ** 2
    return float(np.sqrt(next_var))


@dataclass(frozen=True)
class GarchParams:
    """Fitted GARCH(1,1) coefficients, in the units of the input returns."""

    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    converged: bool

    @property
    def persistence(self) -> float:
        """``alpha + beta``. Must be < 1 for the process to be stationary.

        Daily equity data typically lands around 0.95-0.99: shocks decay, but
        slowly, which is exactly the clustering the unconditional models miss.
        """
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """``omega / (1 - alpha - beta)`` — the level variance reverts to."""
        if self.persistence >= 1.0:
            return float("inf")
        return self.omega / (1.0 - self.persistence)

    @property
    def half_life(self) -> float:
        """Days for a variance shock to decay halfway back to the long-run level."""
        if not 0.0 < self.persistence < 1.0:
            return float("inf")
        return float(np.log(0.5) / np.log(self.persistence))

    def __str__(self) -> str:
        return (
            f"GARCH(1,1) omega={self.omega:.3e} alpha={self.alpha:.4f} "
            f"beta={self.beta:.4f} persistence={self.persistence:.4f} "
            f"half-life={self.half_life:.1f}d"
        )


def _garch_recursion(r: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Variance path ``v_t = omega + alpha * r_{t-1}^2 + beta * v_{t-1}``.

    As with EWMA, ``v_t`` depends only on information available before ``t``.
    """
    n = r.size
    var = np.empty(n)
    persistence = alpha + beta
    # Seed with the model's own long-run variance when stationary, else the
    # sample variance — an unstationary seed would blow the recursion up.
    var[0] = omega / (1.0 - persistence) if persistence < 1.0 else float(np.var(r, ddof=1))
    for t in range(1, n):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
    return var


def _negative_log_likelihood(params: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e10
    var = _garch_recursion(r, omega, alpha, beta)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return 1e10
    # Gaussian quasi-likelihood. Even when the innovations are not normal, the
    # QMLE estimates of (omega, alpha, beta) remain consistent — and the
    # non-normality is handled downstream by the empirical residual quantile.
    return 0.5 * float(np.sum(np.log(2 * np.pi) + np.log(var) + r**2 / var))


def fit_garch11(returns, scale: float = 100.0) -> GarchParams:
    """Fit GARCH(1,1) by (quasi) maximum likelihood.

    Returns are rescaled by ``scale`` before optimising — daily returns are
    ~1e-2, so their squares are ~1e-4 and ``omega`` lands near 1e-6, which is
    numerically awkward for a general-purpose optimiser. Fitting in percent and
    converting back is the standard remedy. The returned parameters are in the
    original units.

    Uses L-BFGS-B with an explicit stationarity constraint enforced in the
    objective, and falls back to a second start point if the first fails to
    converge.
    """
    r = _clean(returns) * scale
    sample_var = float(np.var(r, ddof=1))

    # Start from a typical daily-equity parameterisation: high persistence,
    # small news coefficient.
    starts = [
        np.array([sample_var * 0.05, 0.10, 0.85]),
        np.array([sample_var * 0.20, 0.05, 0.80]),
    ]
    bounds = [(1e-12, None), (0.0, 0.999), (0.0, 0.999)]

    best = None
    for start in starts:
        result = optimize.minimize(
            _negative_log_likelihood,
            start,
            args=(r,),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or result.fun < best.fun:
            best = result
        if result.success:
            break

    # `starts` is non-empty, so the loop always assigns `best`. Asserting it
    # tells the type checker that, and would catch a future edit that empties
    # the start list.
    assert best is not None
    omega, alpha, beta = best.x
    if alpha + beta >= 1.0:
        # Pull just inside the stationary region rather than returning a model
        # whose long-run variance is undefined.
        shrink = 0.999 / (alpha + beta)
        alpha, beta = alpha * shrink, beta * shrink

    # The optimiser worked on rescaled returns, so its log-likelihood is in
    # rescaled units. Converting back via the change-of-variables Jacobian
    # (r_scaled = c * r  =>  ll_original = ll_scaled + n * log c) makes the
    # value directly comparable with any other model fitted to the raw returns.
    log_likelihood = float(-best.fun + r.size * np.log(scale))

    return GarchParams(
        omega=float(omega / scale**2),
        alpha=float(alpha),
        beta=float(beta),
        log_likelihood=log_likelihood,
        converged=bool(best.success),
    )


def garch_variance(returns, params: GarchParams) -> np.ndarray:
    """Conditional variance path implied by fitted parameters."""
    r = _clean(returns)
    return _garch_recursion(r, params.omega, params.alpha, params.beta)


def garch_forecast(returns, params: GarchParams) -> float:
    """One-step-ahead volatility forecast for the period after the sample."""
    r = _clean(returns)
    var = garch_variance(r, params)
    next_var = params.omega + params.alpha * r[-1] ** 2 + params.beta * var[-1]
    return float(np.sqrt(next_var))


def standardised_residuals(returns, variance: np.ndarray) -> np.ndarray:
    """``z_t = r_t / sigma_t`` — the input to Filtered Historical Simulation.

    If the volatility model is doing its job these have unit variance and no
    remaining clustering; whatever fat-tailedness survives is genuine tail
    behaviour rather than a volatility artefact, and is exactly what the
    empirical quantile should capture.
    """
    r = _clean(returns)
    if variance.shape != r.shape:
        raise ValueError(
            f"variance path has shape {variance.shape}, returns have {r.shape}"
        )
    if np.any(variance <= 0):
        raise ValueError("variance path contains non-positive values")
    return r / np.sqrt(variance)
