"""Diversification breakdown — absorption ratio, regimes, and an honest event study.

Why this module exists
----------------------
A portfolio is bought to be diverse and stops being diverse without anyone
changing a position. In stress, assets that were selected precisely because they
behaved differently start moving as one, and the holding count stays flat while
the number of *independent* bets collapses. The pairwise correlation matrix is a
poor instrument for watching that happen: it is N(N-1)/2 numbers moving at once,
with no scalar summary that says how much of the portfolio is now one trade.

The eigenvalue spectrum of the covariance matrix does give that scalar. The
**absorption ratio** (Kritzman, Li, Page & Rigobon, "Principal Components as a
Measure of Systemic Risk", *Journal of Portfolio Management* 37(4), 2011,
pp. 112-126) is the fraction of total variance explained by the top ``K``
eigenvectors:

    AR = sum_{i=1..K} lambda_i / sum_{j=1..N} lambda_j

A high AR means variance has concentrated into few sources — the market is
tightly coupled and a shock has fewer places to dissipate. Their finding, which
this module tries to reproduce out of sample, is that AR *rises before* the
drawdown rather than with it.

Three design points, each of which is where a naive implementation goes wrong:

**The level of AR is not the signal.** AR has a structural level set by ``N`` and
``K``; comparing 0.71 to 0.68 across universes means nothing. The signal is the
standardised *shift* — a short-window mean of AR against a long-window mean,
divided by the long-window standard deviation. That is the source paper's
construction and it is what :func:`standardised_shift` computes.

**The covariance input is a modelling choice, not a detail.** A 500-day rolling
sample covariance averages the trailing two years, so a genuine coupling event
takes months to show up in it. An EWMA conditional covariance
(:func:`ewma_covariance`, sharing the recursion, decay factor and seeding
convention of :mod:`risk_engine.volatility`) reacts to today. Both are selectable
because the difference in lead time between them is itself a result.

**The event list must be fixed before the signal is looked at.** A lead-time
claim measured against events chosen after seeing the signal is unfalsifiable.
:func:`drawdown_events` implements one mechanical rule — peak-to-trough decline
of the equal-weight portfolio beyond a threshold, episodes non-overlapping by
construction — and :func:`event_study` reports false positives alongside hits,
because a hit rate quoted without a false-positive count is not a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .volatility import DEFAULT_BURN_IN, RISKMETRICS_LAMBDA

__all__ = [
    "AbsorptionResult",
    "DrawdownEvent",
    "EventStudyResult",
    "MarkovRegimes",
    "absorption_ratio",
    "drawdown_events",
    "effective_number_of_bets",
    "event_study",
    "ewma_covariance",
    "fit_markov_regimes",
    "realised_vol_signal",
    "rolling_absorption_ratio",
    "sensitivity_grid",
    "standardised_shift",
    "threshold_crossings",
]

#: Kritzman et al.'s convention for the number of retained eigenvectors.
DEFAULT_K_FRACTION = 5


def default_k(n_assets: int) -> int:
    """``floor(N / 5)``, floored at 1 — the convention in the source paper."""
    if n_assets < 1:
        raise ValueError(f"need at least one asset, got {n_assets}")
    return max(1, n_assets // DEFAULT_K_FRACTION)


def _eigenvalues(cov: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of a symmetric covariance matrix, clipped at zero.

    ``eigvalsh`` is used rather than ``eigvals`` because the input is symmetric
    by construction: it is faster, and it returns real values instead of
    complex ones with vanishing imaginary parts. Tiny negative eigenvalues are
    numerical noise from a near-singular sample covariance — clipping them keeps
    the ratio inside [0, 1] without masking a real problem, since a genuinely
    indefinite input would produce a large negative value that survives.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be square, got shape {cov.shape}")
    if not np.isfinite(cov).all():
        raise ValueError("covariance matrix contains NaN or inf")
    values = np.linalg.eigvalsh((cov + cov.T) / 2.0)[::-1]
    return np.clip(values, 0.0, None)


def absorption_ratio(cov: np.ndarray, k: int | None = None) -> float:
    """Fraction of total variance explained by the top ``k`` eigenvectors.

    ``k`` defaults to ``floor(N/5)``. The result lies in ``(0, 1]`` and equals 1
    exactly when ``k == N``.
    """
    values = _eigenvalues(cov)
    n = values.size
    k = default_k(n) if k is None else k
    if not 1 <= k <= n:
        raise ValueError(f"k must lie in [1, {n}], got {k}")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("covariance matrix has zero total variance")
    return float(values[:k].sum() / total)


def effective_number_of_bets(cov: np.ndarray, method: str = "entropy") -> float:
    """How many independent bets the portfolio actually holds.

    The eigenvalues are normalised into a distribution ``p_i = lambda_i / sum``
    and summarised by one of two standard concentration measures:

    ``"entropy"``     ``exp(-sum p_i log p_i)``, the perplexity of the spectrum.
    ``"herfindahl"``  ``1 / sum p_i^2``, the inverse Herfindahl index.

    Both equal ``N`` when every eigenvalue is equal (N genuinely independent
    sources of risk) and fall towards 1 as variance concentrates into one factor.
    Entropy is the default because it responds to the whole spectrum rather than
    being dominated by the largest eigenvalue, which matters here: the interesting
    move is a mid-spectrum collapse, not only the growth of the first component.

    This is the number that makes the finding legible — N holdings, far fewer
    bets — and it is reported alongside AR rather than instead of it because they
    are not redundant: AR fixes a cut at ``k`` while this weighs every eigenvalue.
    """
    values = _eigenvalues(cov)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("covariance matrix has zero total variance")
    p = values / total
    if method == "herfindahl":
        return float(1.0 / np.sum(p**2))
    if method != "entropy":
        raise ValueError(f"unknown method {method!r}")
    # 0 log 0 = 0 by convention; a zero eigenvalue contributes no entropy.
    nonzero = p[p > 0]
    return float(np.exp(-np.sum(nonzero * np.log(nonzero))))


def ewma_covariance(
    returns: pd.DataFrame,
    lam: float = RISKMETRICS_LAMBDA,
    burn_in: int = DEFAULT_BURN_IN,
) -> np.ndarray:
    """Conditional covariance path under an EWMA filter — the matrix analogue of
    :func:`risk_engine.volatility.ewma_variance`.

    The recursion is ``S_t = lam * S_{t-1} + (1 - lam) * r_{t-1} r_{t-1}'``, so
    element ``t`` uses only returns strictly *before* ``t``: it is a one-step-ahead
    forecast, exactly as on the VaR side, and carries no look-ahead.

    Seeding follows the same convention and for the same reason — the sample
    covariance of the first ``burn_in`` rows only, never of the whole series,
    which would leak the future into every element. Elements ``0 .. burn_in - 1``
    are warm-up and callers should discard them; :func:`rolling_absorption_ratio`
    does.

    One caveat specific to the matrix case: the seed block must be long enough
    relative to ``N`` for the sample covariance to be reasonably conditioned. With
    ``burn_in < N`` the seed is singular and the early absorption ratios are
    pinned near 1 by construction. The recursion recovers as the seed decays
    (``lam ** t``), but the warm-up is longer than in the univariate case.

    Returns an array of shape ``(T, N, N)``.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lambda must lie in (0, 1), got {lam}")
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("need a 2-D return frame with at least two assets")
    if not np.isfinite(values).all():
        raise ValueError("return frame contains NaN or inf")
    n_obs, n_assets = values.shape
    if burn_in < 2:
        raise ValueError(f"burn_in must be at least 2, got {burn_in}")
    if n_obs < 2:
        raise ValueError(f"need at least 2 observations, got {n_obs}")

    seed_block = values[: min(burn_in, n_obs)]
    seed = np.cov(seed_block, rowvar=False, ddof=1)
    if not np.isfinite(seed).all() or np.trace(seed) <= 0:
        seed = seed_block.T @ seed_block / max(seed_block.shape[0], 1)

    path = np.empty((n_obs, n_assets, n_assets))
    path[0] = seed
    for t in range(1, n_obs):
        r = values[t - 1]
        path[t] = lam * path[t - 1] + (1.0 - lam) * np.outer(r, r)
    return path


@dataclass(frozen=True)
class AbsorptionResult:
    """A rolling absorption-ratio run and everything needed to interpret it."""

    absorption: pd.Series
    effective_bets: pd.Series
    k: int
    n_assets: int
    window: int
    cov_model: str

    @property
    def baseline(self) -> float:
        """Median AR over the whole run — the 'calm' reference level."""
        return float(self.absorption.median())

    def summary(self) -> str:
        return (
            f"AR[{self.cov_model}] N={self.n_assets} K={self.k} window={self.window} "
            f"median={self.baseline:.4f} min={self.absorption.min():.4f} "
            f"max={self.absorption.max():.4f} "
            f"bets median={self.effective_bets.median():.2f}"
        )


def rolling_absorption_ratio(
    returns: pd.DataFrame,
    window: int = 500,
    k: int | None = None,
    cov_model: str = "sample",
    lam: float = RISKMETRICS_LAMBDA,
    bets_method: str = "entropy",
) -> AbsorptionResult:
    """Absorption ratio and effective number of bets through time.

    Parameters
    ----------
    window
        Rolling estimation window. Kritzman et al. use 500 trading days; shorter
        windows react faster and are noisier, which :func:`sensitivity_grid`
        exists to quantify rather than assert.
    k
        Number of retained eigenvectors, default ``floor(N/5)``.
    cov_model
        ``"sample"`` — plain rolling sample covariance over ``window`` days
        ending at ``t`` inclusive.
        ``"ewma"`` — the conditional covariance of :func:`ewma_covariance`,
        seeded from the first ``window`` observations so both models start
        producing output on the same date and are directly comparable. Note the
        EWMA element at ``t`` is a one-step-ahead forecast using data strictly
        before ``t``, so if anything it is given *less* information than the
        sample branch — the comparison cannot flatter it.

    Both branches are strictly causal; neither can see past ``t``.
    """
    if returns.shape[1] < 2:
        raise ValueError("need at least two assets for an absorption ratio")
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    n_obs, n_assets = returns.shape
    if n_obs <= window:
        raise ValueError(f"need more than {window} observations, got {n_obs}")
    k = default_k(n_assets) if k is None else k
    if not 1 <= k <= n_assets:
        raise ValueError(f"k must lie in [1, {n_assets}], got {k}")

    values = np.asarray(returns, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("return frame contains NaN or inf")

    if cov_model == "sample":
        starts = range(0, n_obs - window + 1)
        covs = [np.cov(values[s : s + window], rowvar=False, ddof=1) for s in starts]
        index = returns.index[window - 1 :]
    elif cov_model == "ewma":
        path = ewma_covariance(returns, lam=lam, burn_in=window)
        covs = list(path[window - 1 :])
        index = returns.index[window - 1 :]
    else:
        raise ValueError(f"unknown cov_model {cov_model!r}")

    ratios = [absorption_ratio(c, k) for c in covs]
    bets = [effective_number_of_bets(c, method=bets_method) for c in covs]
    return AbsorptionResult(
        absorption=pd.Series(ratios, index=index, name="absorption_ratio"),
        effective_bets=pd.Series(bets, index=index, name="effective_bets"),
        k=k,
        n_assets=n_assets,
        window=window,
        cov_model=cov_model,
    )


def standardised_shift(series: pd.Series, short: int = 15, long: int = 250) -> pd.Series:
    """``(short-window mean - long-window mean) / long-window stdev``.

    The source paper's signal construction. Standardising by the long-window
    dispersion is what makes the number comparable across universes, windows and
    ``K`` — and it is why the signal, unlike the level of AR, is **invariant to
    rescaling the returns**: scaling every return by ``c`` scales the covariance
    by ``c^2`` and leaves the eigenvalue *shares* untouched, so AR itself, and
    therefore its standardised shift, does not move at all.

    Both means end at ``t`` inclusive, so the value at ``t`` uses only data
    available at ``t``.
    """
    if short < 1 or long < 2:
        raise ValueError(f"need short >= 1 and long >= 2, got {short}, {long}")
    if short >= long:
        raise ValueError(f"short window ({short}) must be below long ({long})")
    values = pd.Series(series).astype(float)
    short_mean = values.rolling(short).mean()
    long_mean = values.rolling(long).mean()
    long_std = values.rolling(long).std(ddof=1)
    shift = (short_mean - long_mean) / long_std.replace(0.0, np.nan)
    return shift.rename("standardised_shift")


def realised_vol_signal(
    portfolio_returns: pd.Series,
    vol_window: int = 60,
    short: int = 15,
    long: int = 250,
) -> pd.Series:
    """The benchmark trigger: the identical standardised shift, on trailing vol.

    This exists so the absorption result can be falsified. The interesting claim
    is not that a coupling signal precedes drawdowns — trailing volatility does
    that too — but that it moves *first*. Running the same
    :func:`standardised_shift` machinery over trailing realised volatility makes
    the two directly comparable: same transformation, same threshold, same event
    study, different input.
    """
    r = pd.Series(portfolio_returns).astype(float)
    if vol_window < 2:
        raise ValueError(f"vol_window must be at least 2, got {vol_window}")
    vol = r.rolling(vol_window).std(ddof=1) * np.sqrt(252.0)
    return standardised_shift(vol.dropna(), short=short, long=long).rename("vol_signal")


def threshold_crossings(
    signal: pd.Series, threshold: float = 1.0, cooldown: int = 21
) -> pd.DatetimeIndex:
    """Dates on which ``signal`` crosses ``threshold`` from below.

    Only rising edges count: a signal that sits above the threshold for a month
    is one alarm, not twenty, and counting each day separately would inflate both
    the hit rate and the false-positive count without adding information.
    ``cooldown`` suppresses re-triggers within that many observations of the
    previous alarm, so a signal oscillating around the threshold also counts once.
    """
    values = pd.Series(signal).astype(float).dropna()
    if cooldown < 0:
        raise ValueError(f"cooldown must be non-negative, got {cooldown}")
    above = values > threshold
    rising = above & ~above.shift(1, fill_value=False)

    kept: list[pd.Timestamp] = []
    last = -(10**9)
    for position, (stamp, is_rising) in enumerate(rising.items()):
        if is_rising and position - last > cooldown:
            kept.append(stamp)
            last = position
    return pd.DatetimeIndex(kept, name="crossing")


@dataclass(frozen=True)
class DrawdownEvent:
    """One pre-registered drawdown episode: peak, trough, and depth."""

    peak: pd.Timestamp
    trough: pd.Timestamp
    depth: float

    def __str__(self) -> str:
        return (
            f"{self.peak.date()} -> {self.trough.date()}  "
            f"{self.depth:.1%} ({(self.trough - self.peak).days}d)"
        )


def drawdown_events(
    portfolio_returns: pd.Series, min_depth: float = 0.15
) -> list[DrawdownEvent]:
    """Pre-registered event rule. **Fixed before any absorption output is seen.**

    The rule, stated once and applied mechanically:

    1. Compound the portfolio return series into a wealth index.
    2. Track the running maximum. A drawdown episode opens when wealth falls
       below the running maximum and closes when a new maximum is reached (or
       when the series ends).
    3. The episode's depth is its peak-to-trough decline; keep episodes with
       depth ``>= min_depth``.
    4. Onset is the **peak** date — the last day before the decline began, i.e.
       the last moment a warning would still have been actionable.

    Episodes are non-overlapping by construction: a new one cannot open until the
    previous high is recovered, so no calendar day belongs to two events and no
    single decline is double-counted as several.

    ``min_depth`` is a parameter rather than a constant so that the sensitivity
    analysis can vary it, but the study's headline number uses one value chosen
    on the usual "bear market" convention, not on what made the signal look best.
    """
    r = pd.Series(portfolio_returns).astype(float).dropna()
    if r.empty:
        raise ValueError("portfolio return series is empty")
    if not 0.0 < min_depth < 1.0:
        raise ValueError(f"min_depth must lie in (0, 1), got {min_depth}")

    wealth = (1.0 + r).cumprod()
    events: list[DrawdownEvent] = []

    peak_value = float(wealth.iloc[0])
    peak_stamp = wealth.index[0]
    trough_value = peak_value
    trough_stamp = peak_stamp

    for stamp, value in wealth.items():
        value = float(value)
        if value >= peak_value:
            # The previous episode (if any) has now fully recovered; close it.
            depth = (peak_value - trough_value) / peak_value
            if depth >= min_depth:
                events.append(DrawdownEvent(peak_stamp, trough_stamp, depth))
            peak_value, peak_stamp = value, stamp
            trough_value, trough_stamp = value, stamp
        elif value < trough_value:
            trough_value, trough_stamp = value, stamp

    # The series can end inside an unrecovered episode; it still counts.
    depth = (peak_value - trough_value) / peak_value
    if depth >= min_depth:
        events.append(DrawdownEvent(peak_stamp, trough_stamp, depth))
    return events


@dataclass(frozen=True)
class EventStudyResult:
    """Hits, lead times and — mandatorily — false positives."""

    lead_times: list[int]
    n_events: int
    n_signals: int
    false_positives: int
    in_episode: int
    horizon: int
    matched: dict[str, int] = field(default_factory=dict)
    excluded: list[str] = field(default_factory=list)

    @property
    def n_hits(self) -> int:
        return len(self.lead_times)

    @property
    def hit_rate(self) -> float:
        return self.n_hits / self.n_events if self.n_events else 0.0

    @property
    def median_lead(self) -> float:
        return float(np.median(self.lead_times)) if self.lead_times else float("nan")

    @property
    def lead_range(self) -> tuple[int, int]:
        if not self.lead_times:
            return (0, 0)
        return (min(self.lead_times), max(self.lead_times))

    def summary(self) -> str:
        lo, hi = self.lead_range
        skipped = f"  excluded {len(self.excluded)}" if self.excluded else ""
        return (
            f"hits {self.n_hits}/{self.n_events} ({self.hit_rate:.0%}){skipped}  "
            f"median lead {self.median_lead:.0f}d (range {lo}-{hi})  "
            f"signals {self.n_signals}  false positives {self.false_positives}  "
            f"in-episode {self.in_episode}  horizon {self.horizon}d"
        )


def event_study(
    crossings: pd.DatetimeIndex,
    events: list[DrawdownEvent],
    index: pd.DatetimeIndex,
    horizon: int = 60,
    available_from: pd.Timestamp | None = None,
) -> EventStudyResult:
    """Lead time, hit rate and false-positive count for one trigger.

    Lead time is measured in **trading days** — positions in ``index``, not
    calendar days — from the crossing to the event's onset (its peak date). An
    event counts as flagged if some crossing falls in ``[onset - horizon, onset]``;
    the *earliest* qualifying crossing is used, since that is the first warning
    the trigger actually gave.

    Every crossing is then classified, and all three classes are reported:

    * **hit** — followed by an event onset within ``horizon`` trading days;
    * **in-episode** — fired while a drawdown was already under way, so it is
      neither a warning nor a false alarm and is counted separately rather than
      quietly dropped;
    * **false positive** — everything else. This is the number that a hit rate on
      its own conceals, which is why it is not optional here.

    ``available_from`` excludes events whose onset precedes the first date the
    trigger could possibly have fired — with a 500-day covariance window and a
    250-day signal window, the first three years of any sample have no signal at
    all, and scoring those events as misses would understate every trigger by an
    arbitrary amount set by the window lengths. Excluded events are listed rather
    than silently dropped, and comparing two triggers fairly means passing the
    *same* date to both, not each one's own start. Crossings before the cutoff are
    dropped along with the events, for the same reason in reverse: a trigger with
    a shorter warm-up would otherwise accumulate false positives over a stretch
    where its competitor could not fire at all.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    positions = {stamp: i for i, stamp in enumerate(index)}

    excluded: list[str] = []
    if available_from is not None:
        cutoff = pd.Timestamp(available_from)
        evaluable = []
        for event in events:
            if event.peak < cutoff:
                excluded.append(str(event.peak.date()))
            else:
                evaluable.append(event)
        events = evaluable
        crossings = pd.DatetimeIndex([c for c in crossings if c >= cutoff])
    n_events = len(events)

    signal_positions = [positions[s] for s in crossings if s in positions]

    lead_times: list[int] = []
    matched: dict[str, int] = {}
    used: set[int] = set()
    for event in events:
        onset = positions.get(event.peak)
        if onset is None:
            continue
        candidates = [p for p in signal_positions if 0 <= onset - p <= horizon]
        if candidates:
            first = min(candidates)
            lead_times.append(onset - first)
            matched[str(event.peak.date())] = onset - first
            used.update(p for p in candidates)

    # Classify every crossing, so the three classes sum to the signal count.
    episode_spans = [
        (positions[e.peak], positions[e.trough])
        for e in events
        if e.peak in positions and e.trough in positions
    ]
    false_positives = 0
    in_episode = 0
    for p in signal_positions:
        if p in used:
            continue
        if any(start < p <= end for start, end in episode_spans):
            in_episode += 1
        else:
            false_positives += 1

    return EventStudyResult(
        lead_times=lead_times,
        n_events=n_events,
        n_signals=len(signal_positions),
        false_positives=false_positives,
        in_episode=in_episode,
        horizon=horizon,
        matched=matched,
        excluded=excluded,
    )


@dataclass(frozen=True)
class MarkovRegimes:
    """Two-state Markov-switching fit: is 'coupled' a regime or a bad week?"""

    transition_matrix: np.ndarray
    expected_durations: np.ndarray
    state_means: np.ndarray
    smoothed_probabilities: pd.DataFrame
    high_state: int
    log_likelihood: float
    converged: bool

    @property
    def persistence(self) -> np.ndarray:
        """``P[i, i]`` — the probability each state survives another day."""
        return np.diag(self.transition_matrix)

    def summary(self) -> str:
        p = self.transition_matrix
        return (
            f"P=[[{p[0, 0]:.4f}, {p[0, 1]:.4f}], [{p[1, 0]:.4f}, {p[1, 1]:.4f}]]  "
            f"durations={self.expected_durations.round(1).tolist()}d  "
            f"means={self.state_means.round(4).tolist()}  "
            f"high-coupling state={self.high_state}  "
            f"converged={self.converged}"
        )


def fit_markov_regimes(series: pd.Series, switching_variance: bool = True) -> MarkovRegimes:
    """Fit a two-state Markov-switching mean model to an absorption series.

    The purpose is narrow and worth stating: it establishes whether "coupled" is
    a *persistent state* the market sits in for months, or a noisy fortnight that
    a threshold happened to clip. A transition matrix with high diagonal entries
    and a long expected duration is the evidence for the former; without it, the
    threshold-crossing signal would be indistinguishable from chasing noise.

    Expected duration in state ``i`` is ``1 / (1 - P[i, i])``, the mean of the
    geometric distribution of run lengths implied by the chain.

    ``switching_variance`` lets each state carry its own variance, which matters
    because the coupled state is empirically both higher *and* more volatile;
    forcing a common variance would push the fit towards splitting on volatility
    alone.

    **The fit is explicitly seeded**, and that is not cosmetic. An absorption
    series is extremely persistent, and from statsmodels' default EM start the
    likelihood climbs to a degenerate corner — one absorbing state, the other
    with zero variance — which reports a transition matrix of ``[[0, 1], [0, 1]]``
    and means nothing. Starting from persistent transitions (0.98 on the
    diagonal) and the series' own quartiles as the two state means finds a proper
    interior optimum, with a materially higher log-likelihood. The default start
    is kept as a fallback, and ``converged`` is reported rather than assumed.

    Requires ``statsmodels``.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    values = pd.Series(series).astype(float).dropna()
    if values.size < 50:
        raise ValueError(f"need at least 50 observations to fit, got {values.size}")

    data = values.to_numpy()
    model = MarkovRegression(
        data, k_regimes=2, trend="c", switching_variance=switching_variance
    )

    variance = float(np.var(data, ddof=1))
    low, high = (float(np.quantile(data, q)) for q in (0.25, 0.75))
    start = [0.98, 0.02, low, high, variance]
    if switching_variance:
        start.append(variance)

    try:
        fitted = model.fit(start_params=np.array(start), disp=False)
    except Exception:  # noqa: BLE001 — see below
        # The seeded start is a heuristic, and statsmodels signals a bad one by
        # raising from deep inside its EM step (a LinAlgError out of the pinv,
        # among others). Enumerating those is a guess that goes stale; every one
        # of them means the same thing here, which is "use the default start
        # instead". The failure is contained, not hidden — a degenerate result
        # still surfaces through ``converged`` and the transition matrix.
        fitted = model.fit(disp=False)

    transition = np.asarray(fitted.regime_transition).reshape(2, 2)
    # statsmodels stores column-stochastic transitions (P[i, j] = i <- j); the
    # convention everywhere else, including the tests, is row-stochastic.
    transition = transition.T
    diagonal = np.clip(np.diag(transition), 0.0, 1.0 - 1e-12)
    durations = 1.0 / (1.0 - diagonal)

    # The parameter vector leads with the transition probabilities, so the state
    # means must be picked out by name rather than by position.
    means = np.array(
        [float(fitted.params[model.parameters[i, "exog"]][0]) for i in range(2)]
    )
    smoothed = pd.DataFrame(
        np.asarray(fitted.smoothed_marginal_probabilities),
        index=values.index,
        columns=["state_0", "state_1"],
    )
    return MarkovRegimes(
        transition_matrix=transition,
        expected_durations=durations,
        state_means=means,
        smoothed_probabilities=smoothed,
        high_state=int(np.argmax(means)),
        log_likelihood=float(fitted.llf),
        converged=bool(fitted.mle_retvals.get("converged", False)),
    )


def sensitivity_grid(
    returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    windows: list[int],
    ks: list[int | None],
    thresholds: list[float],
    cov_model: str = "ewma",
    min_depth: float = 0.15,
    horizon: int = 60,
    short: int = 15,
    long: int = 250,
    available_from: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Re-run the whole event study across a parameter grid.

    A lead time that exists at exactly one ``(window, K, threshold)`` triple is a
    fit, not a finding. This is the check that distinguishes the two, and it is
    reported whatever it says.
    """
    events = drawdown_events(portfolio_returns, min_depth=min_depth)
    rows = []
    for window in windows:
        for k in ks:
            result = rolling_absorption_ratio(
                returns, window=window, k=k, cov_model=cov_model
            )
            signal = standardised_shift(result.absorption, short=short, long=long)
            for threshold in thresholds:
                crossings = threshold_crossings(signal, threshold=threshold)
                study = event_study(
                    crossings, events, returns.index, horizon=horizon,
                    available_from=available_from,
                )
                rows.append(
                    {
                        "window": window,
                        "k": result.k,
                        "threshold": threshold,
                        "hits": study.n_hits,
                        "events": study.n_events,
                        "hit_rate": study.hit_rate,
                        "median_lead": study.median_lead,
                        "signals": study.n_signals,
                        "false_positives": study.false_positives,
                    }
                )
    return pd.DataFrame(rows)
