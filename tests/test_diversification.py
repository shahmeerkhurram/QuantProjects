"""Diversification / absorption-ratio tests.

The properties asserted here are the ones that separate a real absorption ratio
from a plausible-looking number: the analytic limits (1 for a single factor,
K/N for independent assets), invariance to rescaling, strict causality of both
covariance branches, and — for the event study — recovery of a *planted* signal
whose correct lead time is known by construction.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from risk_engine.diversification import (
    DrawdownEvent,
    absorption_ratio,
    default_k,
    drawdown_events,
    effective_number_of_bets,
    event_study,
    ewma_covariance,
    fit_markov_regimes,
    realised_vol_signal,
    rolling_absorption_ratio,
    sensitivity_grid,
    standardised_shift,
    threshold_crossings,
)
from risk_engine.volatility import ewma_variance


def _frame(values: np.ndarray) -> pd.DataFrame:
    index = pd.bdate_range("2010-01-01", periods=values.shape[0])
    return pd.DataFrame(
        values, index=index, columns=[f"A{i}" for i in range(values.shape[1])]
    )


@pytest.fixture
def independent_returns() -> pd.DataFrame:
    """Ten assets with no common factor — the AR = K/N benchmark case."""
    rng = np.random.default_rng(11)
    return _frame(rng.standard_normal((6000, 10)) * 0.01)


@pytest.fixture
def one_factor_returns() -> pd.DataFrame:
    """Ten assets driven by a single factor plus a whisper of noise."""
    rng = np.random.default_rng(12)
    factor = rng.standard_normal(3000) * 0.01
    noise = rng.standard_normal((3000, 10)) * 1e-5
    return _frame(factor[:, None] + noise)


# --------------------------------------------------------------------------
# The absorption ratio itself
# --------------------------------------------------------------------------


def test_absorption_ratio_lies_in_the_unit_interval(independent_returns):
    cov = np.cov(independent_returns.to_numpy(), rowvar=False)
    for k in range(1, 11):
        ar = absorption_ratio(cov, k)
        assert 0.0 < ar <= 1.0 + 1e-12


def test_absorption_ratio_is_exactly_one_when_k_equals_n(independent_returns):
    """Every eigenvector retained means all variance explained, by definition."""
    cov = np.cov(independent_returns.to_numpy(), rowvar=False)
    assert absorption_ratio(cov, k=10) == pytest.approx(1.0)


def test_absorption_ratio_increases_monotonically_in_k(independent_returns):
    cov = np.cov(independent_returns.to_numpy(), rowvar=False)
    ratios = [absorption_ratio(cov, k) for k in range(1, 11)]
    assert all(b > a for a, b in pairwise(ratios))


def test_absorption_ratio_approaches_one_for_a_single_factor(one_factor_returns):
    """One source of variance absorbs everything, however many assets are held."""
    cov = np.cov(one_factor_returns.to_numpy(), rowvar=False)
    assert absorption_ratio(cov, k=1) > 0.999


def test_absorption_ratio_approaches_k_over_n_for_independent_assets(independent_returns):
    """The null case. With no common factor the spectrum is flat, so the top K
    eigenvalues explain their fair share K/N and nothing more."""
    cov = np.cov(independent_returns.to_numpy(), rowvar=False)
    for k in (2, 5, 8):
        assert absorption_ratio(cov, k) == pytest.approx(k / 10, abs=0.05)


def test_absorption_ratio_is_invariant_to_rescaling_returns(independent_returns):
    """Scaling every return by c scales the covariance by c^2 and leaves the
    eigenvalue *shares* — and therefore AR — untouched."""
    values = independent_returns.to_numpy()
    base = absorption_ratio(np.cov(values, rowvar=False), k=3)
    scaled = absorption_ratio(np.cov(values * 37.0, rowvar=False), k=3)
    assert scaled == pytest.approx(base, rel=1e-10)


def test_default_k_follows_the_paper_convention():
    assert default_k(26) == 5
    assert default_k(10) == 2
    # Never zero: with N < 5 the convention would degenerate to no eigenvectors.
    assert default_k(3) == 1


@pytest.mark.parametrize("bad_k", [0, -1, 11])
def test_absorption_ratio_rejects_k_outside_the_spectrum(independent_returns, bad_k):
    cov = np.cov(independent_returns.to_numpy(), rowvar=False)
    with pytest.raises(ValueError, match="k must lie"):
        absorption_ratio(cov, bad_k)


def test_absorption_ratio_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        absorption_ratio(np.zeros((3, 4)), 1)


def test_absorption_ratio_rejects_a_degenerate_covariance():
    with pytest.raises(ValueError, match="zero total variance"):
        absorption_ratio(np.zeros((3, 3)), 1)


def test_absorption_ratio_rejects_nan():
    cov = np.eye(3)
    cov[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        absorption_ratio(cov, 1)


# --------------------------------------------------------------------------
# Effective number of bets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["entropy", "herfindahl"])
def test_effective_bets_equals_n_for_a_flat_spectrum(method):
    """N equal, independent sources of risk are N genuinely independent bets."""
    assert effective_number_of_bets(np.eye(8), method=method) == pytest.approx(8.0)


@pytest.mark.parametrize("method", ["entropy", "herfindahl"])
def test_effective_bets_collapses_to_one_for_a_single_factor(method, one_factor_returns):
    cov = np.cov(one_factor_returns.to_numpy(), rowvar=False)
    assert effective_number_of_bets(cov, method=method) == pytest.approx(1.0, abs=0.01)


def test_effective_bets_is_bounded_between_one_and_n(independent_returns):
    cov = np.cov(independent_returns.to_numpy(), rowvar=False)
    for method in ("entropy", "herfindahl"):
        bets = effective_number_of_bets(cov, method=method)
        assert 1.0 <= bets <= 10.0 + 1e-9


def test_effective_bets_falls_as_correlation_rises():
    """The measure has to move the right way, or it reports nothing useful."""
    previous = None
    for rho in (0.0, 0.3, 0.6, 0.9):
        cov = np.full((6, 6), rho)
        np.fill_diagonal(cov, 1.0)
        bets = effective_number_of_bets(cov)
        if previous is not None:
            assert bets < previous
        previous = bets


def test_effective_bets_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        effective_number_of_bets(np.eye(3), method="nope")


def test_effective_bets_rejects_a_degenerate_covariance():
    with pytest.raises(ValueError, match="zero total variance"):
        effective_number_of_bets(np.zeros((3, 3)))


# --------------------------------------------------------------------------
# EWMA covariance — reuse of the VaR-side filter
# --------------------------------------------------------------------------


def test_ewma_covariance_diagonal_matches_the_univariate_filter(independent_returns):
    """The matrix filter must *be* the VaR-side filter, not merely resemble it.

    Running the two independently and requiring the diagonal to agree to machine
    precision is what makes 'reuses the EWMA machinery' a checkable claim.
    """
    frame = independent_returns.iloc[:800]
    path = ewma_covariance(frame, burn_in=100)
    for i, column in enumerate(frame.columns):
        univariate = ewma_variance(frame[column].to_numpy(), burn_in=100)
        assert np.allclose(path[:, i, i], univariate, rtol=1e-12, atol=1e-18)


def test_ewma_covariance_has_no_look_ahead(independent_returns):
    """Truncating the input must leave every earlier element unchanged.

    This is the same assertion that caught a real seeding bug in the univariate
    filter; the matrix version inherits the risk and therefore the test.
    """
    frame = independent_returns.iloc[:600]
    full = ewma_covariance(frame, burn_in=50)
    truncated = ewma_covariance(frame.iloc[:400], burn_in=50)
    assert np.allclose(full[:400], truncated, rtol=1e-12, atol=1e-18)


def test_ewma_covariance_is_symmetric_and_positive_semidefinite(independent_returns):
    path = ewma_covariance(independent_returns.iloc[:400], burn_in=60)
    for step in (100, 250, 399):
        cov = path[step]
        assert np.allclose(cov, cov.T, atol=1e-15)
        assert np.linalg.eigvalsh(cov).min() > -1e-12


@pytest.mark.parametrize("lam", [0.0, 1.0, -0.5])
def test_ewma_covariance_rejects_lambda_outside_the_unit_interval(independent_returns, lam):
    with pytest.raises(ValueError, match="lambda"):
        ewma_covariance(independent_returns, lam=lam)


def test_ewma_covariance_rejects_a_single_asset():
    with pytest.raises(ValueError, match="at least two assets"):
        ewma_covariance(pd.DataFrame({"A": [0.01, 0.02, 0.03]}))


def test_ewma_covariance_rejects_nan(independent_returns):
    frame = independent_returns.iloc[:100].copy()
    frame.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        ewma_covariance(frame)


def test_ewma_covariance_rejects_a_short_burn_in(independent_returns):
    with pytest.raises(ValueError, match="burn_in"):
        ewma_covariance(independent_returns, burn_in=1)


# --------------------------------------------------------------------------
# Rolling absorption ratio
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cov_model", ["sample", "ewma"])
def test_rolling_absorption_ratio_is_bounded_and_aligned(independent_returns, cov_model):
    result = rolling_absorption_ratio(
        independent_returns.iloc[:1500], window=250, k=3, cov_model=cov_model
    )
    assert result.k == 3
    assert result.n_assets == 10
    assert len(result.absorption) == 1500 - 250 + 1
    assert result.absorption.index[0] == independent_returns.index[249]
    assert ((result.absorption > 0) & (result.absorption <= 1)).all()
    assert ((result.effective_bets >= 1) & (result.effective_bets <= 10 + 1e-9)).all()
    assert "K=3" in result.summary()


@pytest.mark.parametrize("cov_model", ["sample", "ewma"])
def test_rolling_absorption_ratio_has_no_look_ahead(independent_returns, cov_model):
    """Every value must survive deleting the data that comes after it."""
    frame = independent_returns.iloc[:1200]
    full = rolling_absorption_ratio(frame, window=250, k=3, cov_model=cov_model)
    truncated = rolling_absorption_ratio(
        frame.iloc[:900], window=250, k=3, cov_model=cov_model
    )
    overlap = truncated.absorption.index
    assert np.allclose(
        full.absorption.reindex(overlap).to_numpy(),
        truncated.absorption.to_numpy(),
        rtol=1e-10,
    )


def test_rolling_absorption_ratio_detects_a_planted_coupling_break(independent_returns):
    """A regime that switches from independent to one-factor must show up.

    This is the whole premise of the module, so it is asserted on data whose
    answer is known rather than inferred from market history.
    """
    rng = np.random.default_rng(21)
    calm = rng.standard_normal((800, 10)) * 0.01
    factor = rng.standard_normal(800) * 0.01
    coupled = factor[:, None] + rng.standard_normal((800, 10)) * 0.002
    frame = _frame(np.vstack([calm, coupled]))

    result = rolling_absorption_ratio(frame, window=250, k=2, cov_model="ewma")
    before = result.absorption.iloc[:500].mean()
    after = result.absorption.iloc[-200:].mean()
    assert after > before + 0.3
    # And the diversification count must collapse in the same episode.
    assert result.effective_bets.iloc[-200:].mean() < 2.0


def test_rolling_absorption_ratio_rejects_a_window_longer_than_the_sample(
    independent_returns,
):
    with pytest.raises(ValueError, match="need more than"):
        rolling_absorption_ratio(independent_returns.iloc[:100], window=250)


def test_rolling_absorption_ratio_rejects_an_unknown_covariance_model(independent_returns):
    with pytest.raises(ValueError, match="unknown cov_model"):
        rolling_absorption_ratio(independent_returns, window=250, cov_model="garch")


def test_rolling_absorption_ratio_rejects_a_single_asset(independent_returns):
    with pytest.raises(ValueError, match="at least two assets"):
        rolling_absorption_ratio(independent_returns[["A0"]], window=250)


@pytest.mark.parametrize("bad", [{"window": 1}, {"k": 0}, {"k": 99}])
def test_rolling_absorption_ratio_rejects_invalid_parameters(independent_returns, bad):
    with pytest.raises(ValueError):
        rolling_absorption_ratio(independent_returns.iloc[:600], **bad)


# --------------------------------------------------------------------------
# The signal
# --------------------------------------------------------------------------


def test_standardised_shift_is_invariant_to_rescaling_the_input():
    """A standardised shift is scale-free: it is a z-score, not a difference."""
    rng = np.random.default_rng(31)
    series = pd.Series(
        np.cumsum(rng.standard_normal(1000)) * 0.01 + 5.0,
        index=pd.bdate_range("2010-01-01", periods=1000),
    )
    base = standardised_shift(series, short=15, long=250)
    scaled = standardised_shift(series * 250.0, short=15, long=250)
    assert np.allclose(base.dropna().to_numpy(), scaled.dropna().to_numpy(), rtol=1e-9)


def test_standardised_shift_is_positive_when_the_series_is_rising():
    index = pd.bdate_range("2010-01-01", periods=400)
    rising = pd.Series(np.linspace(0.0, 1.0, 400), index=index)
    shift = standardised_shift(rising, short=15, long=250).dropna()
    assert (shift > 0).all()


def test_standardised_shift_uses_only_past_data():
    index = pd.bdate_range("2010-01-01", periods=600)
    rng = np.random.default_rng(32)
    series = pd.Series(rng.standard_normal(600), index=index)
    full = standardised_shift(series, short=10, long=100)
    truncated = standardised_shift(series.iloc[:400], short=10, long=100)
    assert np.allclose(
        full.iloc[:400].dropna().to_numpy(), truncated.dropna().to_numpy(), rtol=1e-10
    )


@pytest.mark.parametrize("kwargs", [{"short": 0}, {"long": 1}, {"short": 300}])
def test_standardised_shift_rejects_invalid_windows(kwargs):
    series = pd.Series(np.arange(500.0))
    with pytest.raises(ValueError):
        standardised_shift(series, **{"short": 15, "long": 250, **kwargs})


def test_realised_vol_signal_rises_into_a_volatility_burst():
    rng = np.random.default_rng(33)
    calm = rng.standard_normal(900) * 0.005
    burst = rng.standard_normal(120) * 0.05
    pnl = pd.Series(
        np.concatenate([calm, burst]), index=pd.bdate_range("2010-01-01", periods=1020)
    )
    signal = realised_vol_signal(pnl, vol_window=60, short=15, long=250).dropna()
    assert signal.iloc[-1] > 1.0


def test_realised_vol_signal_rejects_a_degenerate_window():
    pnl = pd.Series(np.zeros(500))
    with pytest.raises(ValueError, match="vol_window"):
        realised_vol_signal(pnl, vol_window=1)


def test_threshold_crossings_counts_rising_edges_only():
    """A month spent above the threshold is one alarm, not twenty."""
    index = pd.bdate_range("2010-01-01", periods=60)
    values = np.zeros(60)
    values[10:40] = 2.0        # one sustained excursion
    values[50:55] = 2.0        # a second, after the cooldown
    signal = pd.Series(values, index=index)
    crossings = threshold_crossings(signal, threshold=1.0, cooldown=5)
    assert len(crossings) == 2
    assert crossings[0] == index[10]
    assert crossings[1] == index[50]


def test_threshold_crossings_cooldown_merges_a_flickering_signal():
    index = pd.bdate_range("2010-01-01", periods=40)
    values = np.zeros(40)
    values[[10, 12, 14, 16]] = 2.0   # oscillating around the threshold
    signal = pd.Series(values, index=index)
    assert len(threshold_crossings(signal, 1.0, cooldown=21)) == 1
    assert len(threshold_crossings(signal, 1.0, cooldown=0)) == 4


def test_threshold_crossings_rejects_a_negative_cooldown():
    signal = pd.Series(np.zeros(10), index=pd.bdate_range("2010-01-01", periods=10))
    with pytest.raises(ValueError, match="cooldown"):
        threshold_crossings(signal, 1.0, cooldown=-1)


# --------------------------------------------------------------------------
# Pre-registered events
# --------------------------------------------------------------------------


def test_drawdown_events_recovers_a_planted_decline():
    """A single 20% peak-to-trough decline, placed by construction."""
    index = pd.bdate_range("2010-01-01", periods=300)
    returns = np.zeros(300)
    returns[:100] = 0.001                     # rise to the peak at position 99
    returns[100:150] = np.log1p(-0.20) / 50   # decline into the trough at 149
    returns[150:] = 0.004                     # recovery
    pnl = pd.Series(np.expm1(returns), index=index)

    events = drawdown_events(pnl, min_depth=0.15)
    assert len(events) == 1
    assert events[0].peak == index[99]
    assert events[0].trough == index[149]
    assert events[0].depth == pytest.approx(0.20, abs=0.01)


def test_drawdown_events_are_non_overlapping_and_deep_enough():
    rng = np.random.default_rng(41)
    index = pd.bdate_range("2000-01-01", periods=5000)
    pnl = pd.Series(rng.standard_normal(5000) * 0.012, index=index)
    events = drawdown_events(pnl, min_depth=0.15)
    assert events, "a random walk of this length should produce some drawdowns"
    for event in events:
        assert event.depth >= 0.15
        assert event.peak < event.trough
    for earlier, later in pairwise(events):
        assert earlier.trough < later.peak


def test_drawdown_events_ignores_declines_below_the_threshold():
    index = pd.bdate_range("2010-01-01", periods=200)
    returns = np.zeros(200)
    returns[100:120] = -0.002      # a ~4% dip, far below the rule
    pnl = pd.Series(returns, index=index)
    assert drawdown_events(pnl, min_depth=0.15) == []


def test_drawdown_events_counts_an_unrecovered_decline_at_the_series_end():
    """The sample can end mid-crisis; that episode still happened."""
    index = pd.bdate_range("2010-01-01", periods=200)
    returns = np.zeros(200)
    returns[100:] = np.log1p(-0.30) / 100
    pnl = pd.Series(np.expm1(returns), index=index)
    events = drawdown_events(pnl, min_depth=0.15)
    assert len(events) == 1
    assert events[0].trough == index[-1]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2])
def test_drawdown_events_rejects_an_invalid_depth(bad):
    pnl = pd.Series(np.zeros(100), index=pd.bdate_range("2010-01-01", periods=100))
    with pytest.raises(ValueError, match="min_depth"):
        drawdown_events(pnl, min_depth=bad)


def test_drawdown_events_rejects_an_empty_series():
    with pytest.raises(ValueError, match="empty"):
        drawdown_events(pd.Series(dtype=float))


# --------------------------------------------------------------------------
# The event study
# --------------------------------------------------------------------------


@pytest.fixture
def planted() -> tuple[pd.DatetimeIndex, list[DrawdownEvent]]:
    index = pd.bdate_range("2010-01-01", periods=500)
    events = [DrawdownEvent(index[200], index[260], 0.25)]
    return index, events


def test_event_study_recovers_a_planted_lead_time(planted):
    """The known-truth check: a crossing placed 10 days before onset must be
    reported as a 10-day lead, one hit and no false positives."""
    index, events = planted
    crossings = pd.DatetimeIndex([index[190]])
    result = event_study(crossings, events, index, horizon=60)
    assert result.lead_times == [10]
    assert result.n_hits == 1
    assert result.hit_rate == 1.0
    assert result.median_lead == 10
    assert result.false_positives == 0


def test_event_study_uses_the_earliest_qualifying_crossing(planted):
    """The first warning is the one that counts, not the most flattering one."""
    index, events = planted
    crossings = pd.DatetimeIndex([index[160], index[195]])
    assert event_study(crossings, events, index, horizon=60).lead_times == [40]


def test_event_study_ignores_a_crossing_outside_the_horizon(planted):
    """A warning a year early is not a warning."""
    index, events = planted
    crossings = pd.DatetimeIndex([index[100]])
    result = event_study(crossings, events, index, horizon=60)
    assert result.n_hits == 0
    assert result.hit_rate == 0.0
    assert result.false_positives == 1
    assert np.isnan(result.median_lead)
    assert result.lead_range == (0, 0)


def test_event_study_classifies_every_crossing_exactly_once(planted):
    """Hits, in-episode fires and false positives must partition the crossings —
    otherwise a signal could quietly shed its inconvenient alarms."""
    index, events = planted
    crossings = pd.DatetimeIndex([index[50], index[195], index[230], index[400]])
    result = event_study(crossings, events, index, horizon=60)
    assert result.n_signals == 4
    assert result.n_hits == 1
    assert result.in_episode == 1       # index[230] fired mid-decline
    assert result.false_positives == 2  # index[50] and index[400]


def test_event_study_excludes_events_before_the_signal_could_exist(planted):
    """Scoring an event that predates the signal would penalise every trigger by
    an amount set purely by the window lengths."""
    index, events = planted
    result = event_study(
        pd.DatetimeIndex([index[100], index[400]]), events, index, horizon=60,
        available_from=index[300],
    )
    assert result.n_events == 0
    assert result.excluded == [str(index[200].date())]
    assert result.hit_rate == 0.0
    # The pre-cutoff crossing goes too: a trigger with a shorter warm-up must not
    # collect false positives over a stretch its competitor could not fire in.
    assert result.n_signals == 1
    assert result.false_positives == 1


def test_event_study_summary_reports_false_positives(planted):
    index, events = planted
    summary = event_study(pd.DatetimeIndex([index[190]]), events, index).summary()
    assert "false positives" in summary
    assert "median lead 10d" in summary


def test_event_study_rejects_a_non_positive_horizon(planted):
    index, events = planted
    with pytest.raises(ValueError, match="horizon"):
        event_study(pd.DatetimeIndex([]), events, index, horizon=0)


def test_sensitivity_grid_covers_the_whole_parameter_space(independent_returns):
    pnl = independent_returns.mean(axis=1)
    grid = sensitivity_grid(
        independent_returns.iloc[:1500],
        pnl.iloc[:1500],
        windows=[250, 400],
        ks=[2, None],
        thresholds=[0.5, 1.0],
        min_depth=0.05,
        horizon=40,
        short=10,
        long=100,
    )
    assert len(grid) == 2 * 2 * 2
    assert set(grid.columns) >= {"window", "k", "threshold", "hit_rate", "false_positives"}
    assert (grid["hit_rate"].between(0.0, 1.0)).all()
    assert (grid["false_positives"] >= 0).all()


# --------------------------------------------------------------------------
# Markov switching
# --------------------------------------------------------------------------


@pytest.fixture
def two_regime_series() -> pd.Series:
    """A persistent two-state series with known means, 0.20 and 0.80."""
    rng = np.random.default_rng(51)
    n = 1500
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.99 if state[t - 1] == 0 else 0.98
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    values = np.where(state == 0, 0.20, 0.80) + rng.standard_normal(n) * 0.03
    return pd.Series(values, index=pd.bdate_range("2010-01-01", periods=n))


def test_markov_transition_rows_sum_to_one(two_regime_series):
    regimes = fit_markov_regimes(two_regime_series)
    assert np.allclose(regimes.transition_matrix.sum(axis=1), 1.0, atol=1e-8)
    assert (regimes.transition_matrix >= -1e-12).all()


def test_markov_recovers_planted_state_means(two_regime_series):
    regimes = fit_markov_regimes(two_regime_series)
    means = np.sort(regimes.state_means)
    assert means[0] == pytest.approx(0.20, abs=0.05)
    assert means[1] == pytest.approx(0.80, abs=0.05)


def test_markov_finds_a_persistent_regime_not_a_noisy_week(two_regime_series):
    """The point of the fit: high diagonal persistence and a long duration.

    Without this the threshold signal could not be distinguished from noise, and
    a degenerate fit — which is what the default EM start produces on a series
    this persistent — would sail through unnoticed.
    """
    regimes = fit_markov_regimes(two_regime_series)
    assert (regimes.persistence > 0.90).all()
    assert (regimes.expected_durations > 10).all()
    assert np.allclose(regimes.expected_durations, 1.0 / (1.0 - regimes.persistence))


def test_markov_smoothed_probabilities_are_a_distribution(two_regime_series):
    regimes = fit_markov_regimes(two_regime_series)
    probs = regimes.smoothed_probabilities
    assert len(probs) == len(two_regime_series)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-8)
    assert ((probs >= -1e-9) & (probs <= 1 + 1e-9)).to_numpy().all()


def test_markov_identifies_the_high_coupling_state(two_regime_series):
    regimes = fit_markov_regimes(two_regime_series)
    high = regimes.smoothed_probabilities.iloc[:, regimes.high_state]
    # The high state must be the one that is on when the series is high.
    assert two_regime_series[high > 0.5].mean() > two_regime_series[high <= 0.5].mean()
    assert "high-coupling state" in regimes.summary()


def test_markov_fit_works_without_switching_variance(two_regime_series):
    regimes = fit_markov_regimes(two_regime_series, switching_variance=False)
    assert np.allclose(regimes.transition_matrix.sum(axis=1), 1.0, atol=1e-8)


def test_markov_rejects_too_short_a_series():
    series = pd.Series(np.arange(20.0))
    with pytest.raises(ValueError, match="at least 50"):
        fit_markov_regimes(series)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def test_absorption_figure_draws_both_panels_and_saves(independent_returns, tmp_path):
    from risk_engine.report import plot_absorption_ratio, save_figure

    result = rolling_absorption_ratio(
        independent_returns.iloc[:1200], window=250, k=3, cov_model="ewma"
    )
    signal = standardised_shift(result.absorption, short=15, long=250)
    events = [DrawdownEvent(result.absorption.index[300], result.absorption.index[360], 0.2)]
    fig = plot_absorption_ratio(
        result, events=events, signal=signal,
        crossings=threshold_crossings(signal, 1.0), threshold=1.0,
    )
    assert len(fig.axes) == 2
    out = save_figure(fig, tmp_path / "absorption.png")
    assert out.exists() and out.stat().st_size > 0


def test_absorption_figure_drops_the_signal_panel_when_none_is_given(independent_returns):
    from risk_engine.report import plot_absorption_ratio

    result = rolling_absorption_ratio(independent_returns.iloc[:600], window=250, k=2)
    assert len(plot_absorption_ratio(result).axes) == 1


def test_regime_figure_draws_the_probability_series(two_regime_series, tmp_path):
    from risk_engine.report import plot_regime_probabilities, save_figure

    regimes = fit_markov_regimes(two_regime_series)
    fig = plot_regime_probabilities(regimes, absorption=two_regime_series)
    # The twin axis carrying the absorption series is the second one.
    assert len(fig.axes) == 2
    out = save_figure(fig, tmp_path / "regimes.png")
    assert out.exists() and out.stat().st_size > 0
