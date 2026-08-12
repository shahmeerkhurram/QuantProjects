"""Run the diversification-breakdown event study end to end.

Every number quoted in the README comes from this script. It is deliberately a
script rather than a notebook so the results can be regenerated in one command
and diffed.

    python scripts/diversification_study.py --output outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from risk_engine import (
    drawdown_events,
    event_study,
    fit_markov_regimes,
    load_prices,
    portfolio_returns,
    realised_vol_signal,
    rolling_absorption_ratio,
    sensitivity_grid,
    standardised_shift,
    threshold_crossings,
    to_returns,
)
from risk_engine.report import (
    plot_absorption_ratio,
    plot_regime_probabilities,
    save_figure,
)

UNIVERSE = [
    "AAPL", "MSFT", "INTC", "IBM", "ORCL", "TXN",
    "JPM", "BAC", "GS", "AXP",
    "JNJ", "PFE", "MRK", "UNH",
    "XOM", "CVX", "SLB",
    "PG", "KO", "WMT", "MCD",
    "CAT", "HON", "UNP",
    "NEE", "SO",
]

# Pre-registered study parameters. Fixed before any absorption output was seen;
# see the commit that introduced risk_engine.diversification.
MIN_DEPTH = 0.15      # peak-to-trough decline defining a "drawdown event"
HORIZON = 60          # trading days a warning may precede an onset by
THRESHOLD = 1.0       # standard deviations of the AR shift
SHORT, LONG = 15, 250  # signal windows, as in Kritzman et al.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    prices = load_prices(UNIVERSE, start=args.start, allow_synthetic=False)
    returns = to_returns(prices)
    pnl = portfolio_returns(returns)
    tickers = list(returns.columns)

    print("=" * 78)
    print("1. UNIVERSE")
    print(f"   {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"   {returns.index[0].date()} to {returns.index[-1].date()}  "
          f"({len(returns)} daily observations)")
    print(f"   window={args.window}  K=floor(N/5)  short={SHORT}  long={LONG}")

    ar_ewma = rolling_absorption_ratio(returns, window=args.window, cov_model="ewma")
    ar_sample = rolling_absorption_ratio(returns, window=args.window, cov_model="sample")
    print(f"   K = {ar_ewma.k}")
    print(f"   {ar_ewma.summary()}")
    print(f"   {ar_sample.summary()}")

    events = drawdown_events(pnl, min_depth=MIN_DEPTH)
    print()
    print("=" * 78)
    print(f"PRE-REGISTERED EVENTS (equal-weight, peak-to-trough >= {MIN_DEPTH:.0%})")
    for event in events:
        print(f"   {event}")

    signal = standardised_shift(ar_ewma.absorption, short=SHORT, long=LONG)
    signal_sample = standardised_shift(ar_sample.absorption, short=SHORT, long=LONG)
    vol_signal = realised_vol_signal(pnl, short=SHORT, long=LONG)

    # Every trigger is scored on the same evaluable window — the first date on
    # which the slowest of them could have fired. Otherwise the comparison is
    # decided by window lengths rather than by the signals.
    available_from = max(
        s.first_valid_index() for s in (signal, signal_sample, vol_signal)
    )
    print(f"   signal available from {available_from.date()} "
          f"(shared evaluable window for every trigger)")

    def study_for(sig: pd.Series, threshold: float = THRESHOLD):
        return event_study(
            threshold_crossings(sig, threshold=threshold), events, returns.index,
            horizon=HORIZON, available_from=available_from,
        )

    crossings = threshold_crossings(signal, threshold=THRESHOLD)
    study = study_for(signal)
    study_sample = study_for(signal_sample)
    vol_study = study_for(vol_signal)

    print()
    print("=" * 78)
    print("2. AR LEVEL: calm baseline vs pre-stress peak")
    for name, res in (("ewma", ar_ewma), ("sample", ar_sample)):
        ar = res.absorption
        calm = float(ar.quantile(0.25))
        median = float(ar.median())
        print(f"   [{name}] median {median:.2%}  25th pct (calm) {calm:.2%}  "
              f"max {ar.max():.2%} on {ar.idxmax().date()}")
        for event in events:
            pre = ar.loc[: event.peak]
            if len(pre) < 60:
                continue
            base = float(pre.iloc[-HORIZON - 60: -HORIZON].mean()) if len(pre) > 120 else np.nan
            peak_win = float(pre.iloc[-HORIZON:].max())
            print(f"      {event.peak.date()}: baseline {base:.2%} -> "
                  f"pre-onset peak {peak_win:.2%}")

    print()
    print("=" * 78)
    print("3-5. EVENT STUDY")
    print(f"   excluded (onset before any signal existed): {study.excluded}")
    print(f"   AR (EWMA cov)   : {study.summary()}")
    print(f"     lead times     : {sorted(study.lead_times)}")
    print(f"     matched        : {study.matched}")
    print(f"   AR (sample cov) : {study_sample.summary()}")
    print(f"     lead times     : {sorted(study_sample.lead_times)}")
    print(f"     matched        : {study_sample.matched}")
    print()
    print("8. BENCHMARK — trailing realised volatility, identical machinery")
    print(f"   Realised vol    : {vol_study.summary()}")
    print(f"     lead times     : {sorted(vol_study.lead_times)}")
    print(f"     matched        : {vol_study.matched}")
    print()
    print("   Head to head across thresholds (same events, same horizon):")
    head = []
    for threshold in (0.5, 1.0, 1.5):
        for name, sig in (("absorption_ewma", signal), ("absorption_sample", signal_sample),
                          ("realised_vol", vol_signal)):
            s = study_for(sig, threshold)
            head.append({
                "trigger": name, "threshold": threshold, "hits": s.n_hits,
                "hit_rate": s.hit_rate, "median_lead": s.median_lead,
                "leads": sorted(s.lead_times), "signals": s.n_signals,
                "false_positives": s.false_positives, "in_episode": s.in_episode,
            })
    head_df = pd.DataFrame(head)
    with pd.option_context("display.width", 140):
        print(head_df.to_string(index=False))

    print()
    print("=" * 78)
    print("6. MARKOV SWITCHING (two states, fitted on the EWMA AR series)")
    regimes = fit_markov_regimes(ar_ewma.absorption)
    print(f"   {regimes.summary()}")
    print(f"   transition matrix (row-stochastic):\n{regimes.transition_matrix.round(4)}")
    print(f"   expected durations: {regimes.expected_durations.round(1)}")
    high = regimes.smoothed_probabilities.iloc[:, regimes.high_state]
    print(f"   time in high-coupling state (P>0.5): {(high > 0.5).mean():.1%}")

    print()
    print("=" * 78)
    print("7. EFFECTIVE NUMBER OF BETS")
    bets = ar_ewma.effective_bets
    print(f"   N holdings {len(tickers)}   median {bets.median():.2f}   "
          f"calm (75th pct) {bets.quantile(0.75):.2f}   "
          f"stress (5th pct) {bets.quantile(0.05):.2f}   "
          f"min {bets.min():.2f} on {bets.idxmin().date()}")
    for event in events:
        pre = bets.loc[: event.peak]
        during = bets.loc[event.peak: event.trough]
        if len(pre) and len(during):
            print(f"      {event.peak.date()}: at onset {float(pre.iloc[-1]):.2f} -> "
                  f"trough of episode {float(during.min()):.2f}")

    print()
    print("=" * 78)
    print("SENSITIVITY (EWMA covariance)")
    grid = sensitivity_grid(
        returns, pnl,
        windows=[250, 500, 750],
        ks=[2, 5, 8],
        thresholds=[0.5, 1.0, 1.5],
        cov_model="ewma",
        min_depth=MIN_DEPTH,
        horizon=HORIZON,
        short=SHORT, long=LONG,
        available_from=available_from,
    )
    with pd.option_context("display.width", 120, "display.max_rows", 200):
        print(grid.to_string(index=False))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "absorption_sensitivity.csv", index=False)
    head_df.to_csv(out / "absorption_vs_vol.csv", index=False)
    save_figure(
        plot_absorption_ratio(ar_ewma, events=events, signal=signal,
                              crossings=crossings, threshold=THRESHOLD),
        out / "absorption_ratio.png",
    )
    save_figure(
        plot_regime_probabilities(regimes, absorption=ar_ewma.absorption, events=events),
        out / "absorption_regimes.png",
    )
    summary = {
        "universe": tickers,
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "observations": len(returns),
        "window": args.window,
        "k": ar_ewma.k,
        "events": [
            {"peak": str(e.peak.date()), "trough": str(e.trough.date()), "depth": e.depth}
            for e in events
        ],
        "absorption": {
            "hit_rate": study.hit_rate,
            "hits": study.n_hits,
            "median_lead": study.median_lead,
            "lead_times": study.lead_times,
            "signals": study.n_signals,
            "false_positives": study.false_positives,
            "in_episode": study.in_episode,
        },
        "realised_vol_benchmark": {
            "hit_rate": vol_study.hit_rate,
            "hits": vol_study.n_hits,
            "median_lead": vol_study.median_lead,
            "lead_times": vol_study.lead_times,
            "signals": vol_study.n_signals,
            "false_positives": vol_study.false_positives,
        },
        "markov": {
            "transition_matrix": regimes.transition_matrix.tolist(),
            "expected_durations": regimes.expected_durations.tolist(),
            "state_means": regimes.state_means.tolist(),
            "high_state": regimes.high_state,
        },
        "effective_bets": {
            "n_holdings": len(tickers),
            "median": float(bets.median()),
            "calm_p75": float(bets.quantile(0.75)),
            "stress_p05": float(bets.quantile(0.05)),
            "min": float(bets.min()),
        },
    }
    (out / "diversification_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote figures, sensitivity grid and summary JSON to {out}/")


if __name__ == "__main__":
    main()
