"""Out-of-sample VaR backtesting and the standard regulatory coverage tests.

Counting breaches is not a backtest. A 99% VaR model is expected to be breached
on 1% of days; the questions a risk committee actually asks are

1. *Unconditional coverage* — is the observed breach rate statistically
   distinguishable from 1%?  (Kupiec 1995 proportion-of-failures test)
2. *Independence* — are breaches scattered, or do they arrive in clusters?
   Clustering means the model is blind to volatility regimes even when the
   overall count looks fine.  (Christoffersen 1998)
3. *Conditional coverage* — the joint test of both.

All three are likelihood-ratio tests against a chi-square null. The Basel
traffic-light zone is also reported because it is the classification a bank is
actually supervised against.

**Expected Shortfall needs its own tests.** All three above are tests of a
*quantile*: they look at whether a loss crossed the VaR line and never at how
far past it the loss went. A model can pass every one of them while badly
understating the size of the losses in the tail — which is precisely the
quantity ES exists to report, and the quantity Basel FRTB moved to. Worse, ES is
not elicitable (Gneiting 2011): there is no scoring function it uniquely
minimises, which is why a naive "compare forecast to realisation" backtest does
not exist for it and why the tests below are constructed the way they are.

The Acerbi-Székely (2014) statistics :func:`acerbi_szekely_z1` and
:func:`acerbi_szekely_z2` fill that gap. Neither has a closed-form null, so
:func:`es_backtest` obtains critical values by simulation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .var import RiskResult, conditional_var, historical_var, parametric_var

__all__ = [
    "METHODS",
    "BacktestResult",
    "CoverageTest",
    "ESTest",
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "basel_traffic_light",
    "christoffersen_independence",
    "compare_es_models",
    "compare_models",
    "conditional_coverage",
    "es_backtest",
    "kupiec_pof",
    "rolling_var_backtest",
]

Method = Literal[
    "historical",
    "parametric_normal",
    "parametric_t",
    "ewma_normal",
    "ewma_fhs",
    "garch_fhs",
]

#: Every backtestable model, unconditional first then volatility-filtered.
METHODS: tuple[Method, ...] = (
    "historical",
    "parametric_normal",
    "parametric_t",
    "ewma_normal",
    "ewma_fhs",
    "garch_fhs",
)


@dataclass(frozen=True)
class CoverageTest:
    """Result of a likelihood-ratio coverage test."""

    name: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    reject_at_5pct: bool
    interpretation: str

    def __str__(self) -> str:
        verdict = "REJECT" if self.reject_at_5pct else "pass"
        return f"{self.name:<28} LR={self.statistic:7.3f}  p={self.p_value:6.4f}  [{verdict}]"


@dataclass(frozen=True)
class ESTest:
    """Result of an Acerbi-Székely Expected Shortfall backtest.

    Unlike the coverage tests, the statistic has no closed-form null
    distribution, so ``p_value`` and ``critical_value`` come from simulation and
    ``n_simulations`` is part of the result rather than an implementation
    detail — the number is only as precise as the simulation behind it.
    """

    name: str
    statistic: float
    p_value: float
    critical_value: float
    n_breaches: int
    n_simulations: int
    reject_at_5pct: bool
    interpretation: str

    def __str__(self) -> str:
        verdict = "REJECT" if self.reject_at_5pct else "pass"
        return (
            f"{self.name:<28} Z={self.statistic:+7.4f}  p={self.p_value:6.4f}  "
            f"crit={self.critical_value:+7.4f}  [{verdict}]"
        )


@dataclass
class BacktestResult:
    """Full out-of-sample record for one VaR model."""

    method: str
    confidence: float
    window: int
    var_forecasts: pd.Series
    realised_returns: pd.Series
    breaches: pd.Series
    tests: list[CoverageTest] = field(default_factory=list)
    #: Matching ES forecasts. Optional only for backward compatibility with
    #: results constructed before ES backtesting existed.
    es_forecasts: pd.Series | None = None
    es_tests: list[ESTest] = field(default_factory=list)

    @property
    def n_observations(self) -> int:
        return int(self.breaches.size)

    @property
    def n_breaches(self) -> int:
        return int(self.breaches.sum())

    @property
    def breach_rate(self) -> float:
        return float(self.breaches.mean())

    @property
    def expected_rate(self) -> float:
        return 1.0 - self.confidence

    def summary(self) -> str:
        lines = [
            f"Model            : {self.method}",
            f"Confidence       : {self.confidence:.1%}   rolling window: {self.window}d",
            f"Out-of-sample    : {self.n_observations} days",
            (
                f"Breaches         : {self.n_breaches} "
                f"({self.breach_rate:.2%} observed vs {self.expected_rate:.2%} expected)"
            ),
            (
                "Basel zone       : "
                f"{basel_traffic_light(self.n_breaches, self.n_observations, self.confidence)}"
            ),
            "",
        ]
        lines += [str(t) for t in self.tests]
        return "\n".join(lines)


def kupiec_pof(n_breaches: int, n_obs: int, confidence: float) -> CoverageTest:
    """Kupiec proportion-of-failures test of unconditional coverage.

    Null hypothesis: the true breach probability equals ``1 - confidence``.
    """
    if n_obs <= 0:
        raise ValueError("cannot test an empty backtest")
    p = 1.0 - confidence
    x, n = n_breaches, n_obs
    pi_hat = x / n

    # Log-likelihood under the null (rate fixed at p) and the alternative
    # (rate free at the observed pi). Guard the 0*log(0) boundary cases.
    def _loglik(rate: float) -> float:
        if rate <= 0.0:
            return 0.0 if x == 0 else -np.inf
        if rate >= 1.0:
            return 0.0 if x == n else -np.inf
        return x * np.log(rate) + (n - x) * np.log(1.0 - rate)

    lr = -2.0 * (_loglik(p) - _loglik(pi_hat))
    lr = float(max(lr, 0.0))
    p_value = float(stats.chi2.sf(lr, df=1))
    reject = p_value < 0.05
    if reject:
        direction = "too many" if pi_hat > p else "too few"
        stance = "understating" if pi_hat > p else "overstating"
        note = f"breach rate is {direction} — the model is {stance} risk"
    else:
        note = "breach rate is consistent with the nominal level"
    return CoverageTest("Kupiec (unconditional)", lr, p_value, 1, reject, note)


def christoffersen_independence(breaches) -> CoverageTest:
    """Christoffersen test that breaches are independent across time.

    Null hypothesis: the probability of a breach does not depend on whether
    yesterday was a breach. Rejection means breaches cluster, the signature of a
    model that ignores volatility regimes.
    """
    b = np.asarray(breaches, dtype=int).ravel()
    if b.size < 2:
        raise ValueError("need at least two observations to test independence")

    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total if total else 0.0
    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0

    def _term(count: int, rate: float) -> float:
        # 0 * log(0) is defined as 0 here, the usual convention for this statistic.
        return count * np.log(rate) if count > 0 and rate > 0 else 0.0

    ll_null = _term(n00 + n10, 1 - pi) + _term(n01 + n11, pi)
    ll_alt = _term(n00, 1 - pi0) + _term(n01, pi0) + _term(n10, 1 - pi1) + _term(n11, pi1)
    lr = float(max(-2.0 * (ll_null - ll_alt), 0.0))
    p_value = float(stats.chi2.sf(lr, df=1))
    reject = p_value < 0.05
    note = (
        "breaches cluster in time — the model misses volatility regimes"
        if reject
        else "no evidence of breach clustering"
    )
    return CoverageTest("Christoffersen (independence)", lr, p_value, 1, reject, note)


def conditional_coverage(n_breaches: int, n_obs: int, confidence: float, breaches) -> CoverageTest:
    """Joint test: correct breach rate *and* independent breaches.

    The statistic is the sum of the two component LR statistics, which is
    chi-square with two degrees of freedom under the joint null.
    """
    pof = kupiec_pof(n_breaches, n_obs, confidence)
    ind = christoffersen_independence(breaches)
    lr = pof.statistic + ind.statistic
    p_value = float(stats.chi2.sf(lr, df=2))
    reject = p_value < 0.05
    note = (
        "joint null rejected — model fails coverage, clustering, or both"
        if reject
        else "model passes joint conditional coverage"
    )
    return CoverageTest("Conditional coverage (joint)", lr, p_value, 2, reject, note)


def basel_zone_thresholds(n_obs: int, confidence: float = 0.99) -> tuple[int, int]:
    """Largest breach counts still in the green and amber zones.

    The published Basel table (green 0-4, amber 5-9, red 10+) is not arbitrary:
    it is derived from the binomial distribution of breach counts. The green
    zone ends where the cumulative probability first reaches 95%, and the amber
    zone ends where it reaches 99.99%.

    Deriving the thresholds rather than hardcoding them means they stay correct
    for any sample length *and* any confidence level. Naively rescaling the 4/9
    figures — as an earlier version of this function did — is wrong twice over:
    the binomial is not linear in ``n``, and at 95% confidence you expect 12.5
    breaches per 250 days, so a "red at 10" verdict would be nonsense.

    Sanity check: ``basel_zone_thresholds(250, 0.99) == (4, 9)``, reproducing
    the supervisory table exactly.
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}")
    p = 1.0 - confidence
    counts = np.arange(0, n_obs + 1)
    cdf = stats.binom.cdf(counts, n_obs, p)
    green_max = int(np.searchsorted(cdf, 0.95, side="left")) - 1
    amber_max = int(np.searchsorted(cdf, 0.9999, side="left")) - 1
    return max(green_max, 0), max(amber_max, green_max)


def basel_traffic_light(n_breaches: int, n_obs: int, confidence: float = 0.99) -> str:
    """Basel traffic-light zone for an observed breach count.

    Thresholds come from :func:`basel_zone_thresholds`, so the classification is
    valid at any sample length and confidence level, not only the supervisory
    250-day / 99% case.
    """
    if n_obs <= 0:
        return "n/a"
    green_max, amber_max = basel_zone_thresholds(n_obs, confidence)
    if n_breaches <= green_max:
        return f"GREEN ({n_breaches} breaches)"
    if n_breaches <= amber_max:
        return f"AMBER ({n_breaches} breaches) — model under review"
    return f"RED ({n_breaches} breaches) — model rejected"


def _standardised_t_tail(df: float, alpha: float) -> tuple[float, float]:
    """``(VaR, ES)`` at level ``alpha`` for a unit-variance Student-t.

    Both are positive loss magnitudes, matching the convention used throughout
    the engine. The closed form for the t's expected shortfall is

        ES = f(q) / alpha * (df + q^2) / (df - 1)

    in *standard* t units, where ``q`` is the alpha-quantile and ``f`` its
    density; the scaling by ``sqrt((df - 2) / df)`` converts to unit variance.
    """
    if df <= 2.0:
        raise ValueError(f"need df > 2 for a finite variance, got {df}")
    scale = np.sqrt((df - 2.0) / df)
    q = stats.t.ppf(alpha, df)
    density = stats.t.pdf(q, df)
    es_standard = density / alpha * (df + q**2) / (df - 1.0)
    return float(-q * scale), float(es_standard * scale)


def _null_shape(es_var_ratio: float, alpha: float) -> float:
    """Degrees of freedom whose ES/VaR ratio matches the model's own.

    The null hypothesis of an ES backtest is that *the model's predictive
    distribution is correct*. Simulating that null needs the whole distribution,
    but a walk-forward backtest only records two numbers per day — the VaR and
    the ES. Those two pin down the tail shape within a one-parameter family, and
    a Student-t is the natural choice: it spans the range from a normal tail
    (``df -> inf``) to a very heavy one, monotonically in ``df``.

    Matching the ratio matters more than it might appear. Simulating from a
    normal null when the model claims a fat tail would reject the model for
    being right, since its ES would exceed what a normal tail can produce.

    Returns ``inf`` when the requested ratio is at or below the normal one,
    which is then handled by simulating from a normal.
    """
    normal_ratio = _normal_tail(alpha)[1] / _normal_tail(alpha)[0]
    if not np.isfinite(es_var_ratio) or es_var_ratio <= normal_ratio:
        return float("inf")

    def gap(df: float) -> float:
        var, es = _standardised_t_tail(df, alpha)
        return es / var - es_var_ratio

    # The ratio falls monotonically in df, so gap(low) > 0 > gap(high) brackets
    # the root. If even df = 2.05 cannot produce a tail as heavy as the model
    # claims, the heaviest admissible one is returned rather than raising: an
    # unbracketable ratio is a statement about the model, not a solver failure.
    low, high = 2.05, 500.0
    if gap(low) < 0:
        return low
    try:
        return float(optimize.brentq(gap, low, high, xtol=1e-6))
    except ValueError:
        return float("inf")


def _normal_tail(alpha: float) -> tuple[float, float]:
    """``(VaR, ES)`` at level ``alpha`` for a standard normal."""
    q = stats.norm.ppf(alpha)
    return float(-q), float(stats.norm.pdf(q) / alpha)


def acerbi_szekely_z1(
    losses: np.ndarray, var_forecasts: np.ndarray, es_forecasts: np.ndarray
) -> float:
    """Acerbi-Székely Test 1 — is the ES right *given* a breach happened?

        Z1 = 1 - mean over breach days of (loss / ES)

    Conditional on the breach set, so it says nothing about whether breaches
    arrive at the right rate — that is what the coverage tests are for. Zero
    under a correctly specified model; **negative means the tail is understated**,
    which is the direction that matters.

    Returns NaN when there are no breaches, since the statistic is a mean over
    an empty set. That is reported rather than silently treated as a pass.

    Acerbi & Székely (2014), "Back-testing Expected Shortfall", Risk 27(11).
    """
    breaches = losses > var_forecasts
    n = int(breaches.sum())
    if n == 0:
        return float("nan")
    return float(1.0 - np.mean(losses[breaches] / es_forecasts[breaches]))


def acerbi_szekely_z2(
    losses: np.ndarray,
    var_forecasts: np.ndarray,
    es_forecasts: np.ndarray,
    confidence: float,
) -> float:
    """Acerbi-Székely Test 2 — the unconditional test.

        Z2 = 1 - (1 / (T * alpha)) * sum over breach days of (loss / ES)

    Unlike Test 1 this does not condition on the breach count, so it tests the
    frequency *and* the magnitude of tail losses jointly: a model can fail it by
    breaching too often at the right average severity, or the right number of
    times too severely. Zero in expectation under a correct model, negative when
    risk is understated.

    Acerbi & Székely (2014), "Back-testing Expected Shortfall", Risk 27(11).
    """
    alpha = 1.0 - confidence
    breaches = losses > var_forecasts
    total = float(np.sum(losses[breaches] / es_forecasts[breaches]))
    return float(1.0 - total / (losses.size * alpha))


def es_backtest(
    result: BacktestResult,
    n_simulations: int = 10_000,
    seed: int = 20140601,
    test: str = "z2",
) -> ESTest:
    """Test an ES forecast series by simulating its own null distribution.

    Neither Z1 nor Z2 has a closed-form null, so critical values are obtained by
    simulation: draw ``n_simulations`` synthetic return paths from the model's
    own predictive distribution, recompute the statistic on each, and read off
    the 5th percentile. The test is one-sided — only understatement of risk is a
    supervisory failure, and it is the left tail that detects it.

    The predictive distribution is reconstructed from what the backtest records.
    See :func:`_null_shape` for why the ES/VaR ratio is matched rather than a
    normal null being assumed; the daily scale is then set so that each day's
    simulated VaR reproduces the model's own forecast. The reconstruction is an
    approximation to Acerbi and Székely's procedure, which assumes the full
    predictive distribution is available — it is stated here rather than hidden
    because it is the main caveat on the p-values.
    """
    if result.es_forecasts is None:
        raise ValueError(
            f"backtest for {result.method!r} carries no ES forecasts; "
            "rerun rolling_var_backtest to record them"
        )
    if n_simulations < 100:
        raise ValueError(f"need at least 100 simulations, got {n_simulations}")

    losses = -result.realised_returns.to_numpy(dtype=float)
    var = result.var_forecasts.to_numpy(dtype=float)
    es = result.es_forecasts.to_numpy(dtype=float)
    if np.any(es <= 0) or np.any(var <= 0):
        raise ValueError("VaR and ES forecasts must be strictly positive")

    alpha = 1.0 - result.confidence
    statistic = (
        acerbi_szekely_z1(losses, var, es)
        if test == "z1"
        else acerbi_szekely_z2(losses, var, es, result.confidence)
    )
    if test not in ("z1", "z2"):
        raise ValueError(f"unknown test {test!r}; use 'z1' or 'z2'")

    # Reconstruct the null: a tail shape matching the model's own ES/VaR ratio,
    # scaled daily to reproduce the model's VaR.
    df = _null_shape(float(np.mean(es / var)), alpha)
    rng = np.random.default_rng(seed)
    n_obs = losses.size
    if np.isfinite(df):
        var_unit, _ = _standardised_t_tail(df, alpha)
        draws = rng.standard_t(df, size=(n_simulations, n_obs)) * np.sqrt(
            (df - 2.0) / df
        )
    else:
        var_unit, _ = _normal_tail(alpha)
        draws = rng.standard_normal((n_simulations, n_obs))

    # sigma_t chosen so the simulated day-t alpha-quantile equals VaR_t.
    sigma = var / var_unit
    simulated_losses = draws * sigma  # symmetric, so the sign convention is free

    if test == "z1":
        hits = simulated_losses > var
        counts = hits.sum(axis=1)
        ratios = np.where(hits, simulated_losses / es, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            null = 1.0 - ratios / counts
        null = null[counts > 0]
    else:
        hits = simulated_losses > var
        ratios = np.where(hits, simulated_losses / es, 0.0).sum(axis=1)
        null = 1.0 - ratios / (n_obs * alpha)

    if null.size == 0 or not np.isfinite(statistic):
        return ESTest(
            name=f"Acerbi-Székely {test.upper()}",
            statistic=statistic,
            p_value=float("nan"),
            critical_value=float("nan"),
            n_breaches=int((losses > var).sum()),
            n_simulations=n_simulations,
            reject_at_5pct=False,
            interpretation="no breaches — statistic undefined, model untestable here",
        )

    critical = float(np.quantile(null, 0.05))
    p_value = float(np.mean(null <= statistic))
    reject = bool(statistic < critical)
    if reject:
        interpretation = "tail losses exceed the forecast ES — risk understated"
    elif statistic > 0:
        interpretation = "realised tail losses milder than forecast — ES conservative"
    else:
        interpretation = "ES consistent with realised tail losses"

    return ESTest(
        name=f"Acerbi-Székely {test.upper()}",
        statistic=statistic,
        p_value=p_value,
        critical_value=critical,
        n_breaches=int((losses > var).sum()),
        n_simulations=n_simulations,
        reject_at_5pct=reject,
        interpretation=interpretation,
    )


def _forecaster(method: Method, confidence: float) -> Callable[[np.ndarray], RiskResult]:
    """Map a method name to a function of one window of returns.

    GARCH is absent here because it needs state across calls (the cached fit);
    :func:`rolling_var_backtest` handles it separately.
    """
    if method == "historical":
        return lambda w: historical_var(w, confidence)
    if method == "parametric_normal":
        return lambda w: parametric_var(w, confidence, distribution="normal")
    if method == "parametric_t":
        return lambda w: parametric_var(w, confidence, distribution="student_t")
    if method == "ewma_normal":
        return lambda w: conditional_var(w, confidence, model="ewma", innovation="normal")
    if method == "ewma_fhs":
        return lambda w: conditional_var(w, confidence, model="ewma", innovation="empirical")
    raise ValueError(f"unknown backtest method {method!r}")


def rolling_var_backtest(
    returns: pd.Series,
    confidence: float = 0.99,
    window: int = 250,
    method: Method = "historical",
    refit_every: int = 25,
) -> BacktestResult:
    """Walk-forward backtest: forecast tomorrow's VaR using only past data.

    Each forecast at date ``t`` is estimated from the ``window`` observations
    strictly before ``t``, so there is no look-ahead. This is the only way a VaR
    number means anything — an in-sample VaR is guaranteed to have the right
    breach count and tests nothing.

    Parameters
    ----------
    refit_every
        GARCH only: how often to re-run the maximum-likelihood fit. Refitting on
        every one of ~1,900 steps would dominate runtime for no material gain,
        since the parameters barely move day to day. The *variance recursion*
        still updates every step from the newest return — only the coefficients
        are held fixed between refits, so there is still no look-ahead.
    """
    series = pd.Series(returns).dropna()
    if window < 30:
        raise ValueError(f"window={window} is too short for a stable tail estimate")
    if series.size <= window:
        raise ValueError(
            f"need more than {window} observations to backtest, got {series.size}"
        )
    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1, got {refit_every}")

    values = series.to_numpy(dtype=float)
    forecasts = np.empty(series.size - window)
    # The estimators return ES alongside VaR at no extra cost; recording it is
    # what makes an Expected Shortfall backtest possible at all.
    es_forecasts = np.empty(series.size - window)

    if method == "garch_fhs":
        from .volatility import fit_garch11

        params = None
        for i in range(window, series.size):
            step = i - window
            if step % refit_every == 0:
                params = fit_garch11(values[i - window : i])
            estimated = conditional_var(
                values[i - window : i],
                confidence,
                model="garch",
                innovation="empirical",
                garch_params=params,
            )
            forecasts[step] = estimated.var
            es_forecasts[step] = estimated.expected_shortfall
    else:
        estimate = _forecaster(method, confidence)
        for i in range(window, series.size):
            estimated = estimate(values[i - window : i])
            forecasts[i - window] = estimated.var
            es_forecasts[i - window] = estimated.expected_shortfall

    index = series.index[window:]
    realised = series.iloc[window:]
    var_series = pd.Series(forecasts, index=index, name=f"VaR_{method}")
    es_series = pd.Series(es_forecasts, index=index, name=f"ES_{method}")
    # A breach is a realised loss strictly worse than the forecast loss level.
    breaches = (-realised > var_series).astype(int)
    breaches.name = "breach"

    result = BacktestResult(
        method=method,
        confidence=confidence,
        window=window,
        var_forecasts=var_series,
        realised_returns=realised,
        breaches=breaches,
        es_forecasts=es_series,
    )
    result.tests = [
        kupiec_pof(result.n_breaches, result.n_observations, confidence),
        christoffersen_independence(breaches.to_numpy()),
        conditional_coverage(
            result.n_breaches, result.n_observations, confidence, breaches.to_numpy()
        ),
    ]
    return result


def compare_models(
    returns: pd.Series,
    confidence: float = 0.99,
    window: int = 250,
    methods: tuple[Method, ...] = METHODS,
    refit_every: int = 25,
) -> pd.DataFrame:
    """Backtest every model and tabulate the verdicts side by side.

    This is the table that answers "which model should we actually use?". Sorted
    by the joint conditional-coverage p-value, descending — the highest p-value
    is the model the data argues against least.
    """
    rows = []
    for method in methods:
        bt = rolling_var_backtest(returns, confidence, window, method, refit_every)
        by_name = {t.name: t for t in bt.tests}
        kupiec = by_name["Kupiec (unconditional)"]
        christo = by_name["Christoffersen (independence)"]
        joint = by_name["Conditional coverage (joint)"]
        rows.append(
            {
                "method": method,
                "breaches": bt.n_breaches,
                "breach_rate": bt.breach_rate,
                "expected_rate": bt.expected_rate,
                "kupiec_p": kupiec.p_value,
                "christoffersen_p": christo.p_value,
                "joint_p": joint.p_value,
                "passes_all": not any(t.reject_at_5pct for t in bt.tests),
                "basel": basel_traffic_light(
                    bt.n_breaches, bt.n_observations, confidence
                ).split(" ")[0],
                "mean_var": float(bt.var_forecasts.mean()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("joint_p", ascending=False)
        .reset_index(drop=True)
    )


def compare_es_models(
    returns: pd.Series,
    confidence: float = 0.975,
    window: int = 250,
    methods: tuple[Method, ...] = METHODS,
    refit_every: int = 25,
    n_simulations: int = 10_000,
    seed: int = 20140601,
) -> pd.DataFrame:
    """Run every model through both ES backtests alongside the VaR coverage tests.

    Defaults to 97.5% because that is the level Basel FRTB moved Expected
    Shortfall to, and because the whole point of the comparison is whether the
    VaR verdict and the ES verdict agree. Both are reported in one table so that
    a model passing one and failing the other is visible rather than needing to
    be reconstructed from two.
    """
    rows = []
    for method in methods:
        bt = rolling_var_backtest(returns, confidence, window, method, refit_every)
        z1 = es_backtest(bt, n_simulations=n_simulations, seed=seed, test="z1")
        z2 = es_backtest(bt, n_simulations=n_simulations, seed=seed, test="z2")
        bt.es_tests = [z1, z2]
        # rolling_var_backtest always records ES; assert it for the type checker
        # rather than defaulting, which would hide a real regression.
        assert bt.es_forecasts is not None
        by_name = {t.name: t for t in bt.tests}
        rows.append(
            {
                "method": method,
                "breaches": bt.n_breaches,
                "breach_rate": bt.breach_rate,
                "mean_var": float(bt.var_forecasts.mean()),
                "mean_es": float(bt.es_forecasts.mean()),
                "z1": z1.statistic,
                "z1_p": z1.p_value,
                "z1_pass": not z1.reject_at_5pct,
                "z2": z2.statistic,
                "z2_p": z2.p_value,
                "z2_pass": not z2.reject_at_5pct,
                "kupiec_p": by_name["Kupiec (unconditional)"].p_value,
                "christoffersen_p": by_name["Christoffersen (independence)"].p_value,
                "var_passes_all": not any(t.reject_at_5pct for t in bt.tests),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("z2_p", ascending=False)
        .reset_index(drop=True)
    )
