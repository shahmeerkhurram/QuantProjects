"""Export engine results as JSON for the portfolio site to consume.

The website used to carry these numbers as hand-typed literals, which meant that
every rerun of the engine silently invalidated the published page. This script
makes the engine the single source of truth: run it, and the site picks up
whatever the engine currently says.

Usage
-----
    python scripts/export_site_data.py --site ../shahmeerkhurram.github.io

Writes ``src/data/risk-engine-results.json`` and
``src/data/diversification-results.json``, and copies the charts into
``public/risk-engine/`` and ``public/diversification/``.

The diversification payload is deliberately produced by the same study code the
README and the handoff report quote, so the site, the repository and the write-up
cannot disagree about a number.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from risk_engine import (
    __version__,
    conditional_var,
    correlation_network,
    drawdown_events,
    event_study,
    fit_markov_regimes,
    historical_var,
    load_prices,
    monte_carlo_var,
    parametric_var,
    portfolio_returns,
    rank_systemic_assets,
    realised_vol_signal,
    rolling_absorption_ratio,
    sensitivity_grid,
    standardised_shift,
    threshold_crossings,
    to_returns,
)
from risk_engine.backtest import compare_models

TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
NETWORK_TICKERS = [*TICKERS, "JNJ", "WMT"]


def _count_tests(repo_root: Path) -> int | None:
    """Ask pytest how many tests exist, so the site never quotes a stale count."""
    try:
        proc = subprocess.run(
            # sys.executable, not "python": the latter may resolve to a
            # different interpreter with no pytest installed.
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    # pytest's --collect-only -q output format varies by version. Newer
    # versions print one "path/to/test_x.py: N" line per file with no summary;
    # older ones print a single "N tests collected" line. Handle both rather
    # than silently reporting None.
    per_file = re.findall(r"^\S+\.py:\s*(\d+)\s*$", proc.stdout, flags=re.MULTILINE)
    if per_file:
        return sum(int(n) for n in per_file)

    summary = re.search(r"(\d+)\s+tests?\s+collected", proc.stdout)
    return int(summary.group(1)) if summary else None


def build_payload(repo_root: Path) -> dict:
    prices = load_prices(TICKERS, start="2018-01-01")
    returns = to_returns(prices)
    pnl = portfolio_returns(returns)

    estimates = [
        historical_var(pnl, 0.99),
        parametric_var(pnl, 0.99, distribution="normal"),
        parametric_var(pnl, 0.99, distribution="student_t"),
        monte_carlo_var(returns, confidence=0.99, engine="gaussian"),
        monte_carlo_var(returns, confidence=0.99, engine="student_t"),
        monte_carlo_var(returns, confidence=0.99, engine="bootstrap"),
        conditional_var(pnl, 0.99, model="ewma", innovation="empirical"),
        conditional_var(pnl, 0.99, model="garch", innovation="empirical"),
    ]

    backtests = {
        f"{level}": compare_models(pnl, level, 250).to_dict("records")
        for level in (0.99, 0.975)
    }

    network_returns = to_returns(load_prices(NETWORK_TICKERS, start="2018-01-01"))
    graph = correlation_network(network_returns, method="threshold", threshold=0.35)
    systemic = rank_systemic_assets(graph, initial_distress=0.30).to_dict("records")

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "engine_version": __version__,
        "test_count": _count_tests(repo_root),
        "universe": TICKERS,
        "observations": len(pnl),
        "period": {
            "start": str(pnl.index[0].date()),
            "end": str(pnl.index[-1].date()),
        },
        "var_estimates": [r.as_row() for r in estimates],
        "backtests": backtests,
        "systemic": systemic,
    }


def build_diversification_payload() -> dict:
    """Re-run the diversification study and serialise what the site page needs.

    The universe, window and pre-registered event rule are imported from
    ``diversification_study`` rather than restated here: two copies of a
    pre-registered rule is one copy too many, and the whole claim rests on that
    rule being fixed.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from diversification_study import HORIZON, LONG, MIN_DEPTH, SHORT, THRESHOLD, UNIVERSE

    returns = to_returns(load_prices(UNIVERSE, start="2007-01-01"))
    pnl = portfolio_returns(returns)

    ewma = rolling_absorption_ratio(returns, window=500, cov_model="ewma")
    sample = rolling_absorption_ratio(returns, window=500, cov_model="sample")
    signal = standardised_shift(ewma.absorption, short=SHORT, long=LONG)
    signal_sample = standardised_shift(sample.absorption, short=SHORT, long=LONG)
    vol_signal = realised_vol_signal(pnl, short=SHORT, long=LONG)
    events = drawdown_events(pnl, min_depth=MIN_DEPTH)

    # Every trigger is scored from the first date the slowest of them could have
    # fired, so the comparison is decided by the signals and not their warm-ups.
    available_from = max(
        s.first_valid_index() for s in (signal, signal_sample, vol_signal)
    )

    def study(series, threshold: float):
        return event_study(
            threshold_crossings(series, threshold=threshold), events, returns.index,
            horizon=HORIZON, available_from=available_from,
        )

    def row(name: str, series, threshold: float) -> dict:
        result = study(series, threshold)
        return {
            "trigger": name,
            "threshold": threshold,
            "hits": result.n_hits,
            "events": result.n_events,
            "hit_rate": result.hit_rate,
            "median_lead": None if not result.lead_times else result.median_lead,
            "lead_times": sorted(result.lead_times),
            "signals": result.n_signals,
            "false_positives": result.false_positives,
            "in_episode": result.in_episode,
        }

    triggers = [
        row(name, series, threshold)
        for threshold in (0.5, 1.0, 1.5)
        for name, series in (
            ("absorption_ewma", signal),
            ("absorption_sample", signal_sample),
            ("realised_vol", vol_signal),
        )
    ]

    regimes = fit_markov_regimes(ewma.absorption)
    grid = sensitivity_grid(
        returns, pnl, windows=[250, 500, 750], ks=[2, 5, 8],
        thresholds=[0.5, 1.0, 1.5], cov_model="ewma", min_depth=MIN_DEPTH,
        horizon=HORIZON, short=SHORT, long=LONG, available_from=available_from,
    )
    headline = study(signal, THRESHOLD)
    vol_headline = study(vol_signal, THRESHOLD)
    bets = ewma.effective_bets

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "universe": list(returns.columns),
        "n_assets": ewma.n_assets,
        "observations": len(returns),
        "period": {
            "start": str(returns.index[0].date()),
            "end": str(returns.index[-1].date()),
        },
        "parameters": {
            "window": ewma.window,
            "k": ewma.k,
            "short": SHORT,
            "long": LONG,
            "threshold": THRESHOLD,
            "min_depth": MIN_DEPTH,
            "horizon": HORIZON,
            "cov_model": ewma.cov_model,
            "evaluable_from": str(available_from.date()),
        },
        "absorption": {
            "median": float(ewma.absorption.median()),
            "calm_p25": float(ewma.absorption.quantile(0.25)),
            "peak": float(ewma.absorption.max()),
            "peak_date": str(ewma.absorption.idxmax().date()),
            "sample_median": float(sample.absorption.median()),
        },
        "effective_bets": {
            "n_holdings": ewma.n_assets,
            "median": float(bets.median()),
            "calm_p75": float(bets.quantile(0.75)),
            "stress_p05": float(bets.quantile(0.05)),
            "min": float(bets.min()),
            "min_date": str(bets.idxmin().date()),
        },
        "events": [
            {
                "peak": str(e.peak.date()),
                "trough": str(e.trough.date()),
                "depth": e.depth,
                "evaluable": e.peak >= available_from,
            }
            for e in events
        ],
        "headline": {
            "absorption": row("absorption_ewma", signal, THRESHOLD),
            "realised_vol": row("realised_vol", vol_signal, THRESHOLD),
            "excluded_events": headline.excluded,
            "absorption_beats_vol": (
                bool(headline.lead_times)
                and bool(vol_headline.lead_times)
                and headline.median_lead > vol_headline.median_lead
            ),
        },
        "triggers": triggers,
        "sensitivity": grid.replace({float("nan"): None}).to_dict("records"),
        "markov": {
            "transition_matrix": regimes.transition_matrix.tolist(),
            "expected_durations": regimes.expected_durations.tolist(),
            "state_means": regimes.state_means.tolist(),
            "high_state": regimes.high_state,
            "converged": regimes.converged,
            "share_high": float(
                (regimes.smoothed_probabilities.iloc[:, regimes.high_state] > 0.5).mean()
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, help="path to the Astro site repo")
    parser.add_argument("--outputs", default="outputs", help="directory holding charts")
    parser.add_argument("--skip-diversification", action="store_true",
                        help="skip the diversification study (it is the slow part)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    site = Path(args.site).resolve()
    if not (site / "src" / "data").is_dir():
        raise SystemExit(f"error: {site} does not look like the site repo")

    payload = build_payload(repo_root)
    target = site / "src" / "data" / "risk-engine-results.json"
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {target}")

    if not args.skip_diversification:
        diversification = build_diversification_payload()
        target = site / "src" / "data" / "diversification-results.json"
        target.write_text(json.dumps(diversification, indent=2, default=str) + "\n")
        print(f"wrote {target}")

    charts = Path(args.outputs)
    if charts.is_dir():
        destination = site / "public" / "risk-engine"
        destination.mkdir(parents=True, exist_ok=True)
        for png in sorted(charts.glob("*.png")):
            shutil.copy2(png, destination / png.name)
            print(f"copied {png.name}")
    else:
        print(f"note: {charts} not found — run the CLI with --output first")

    # The absorption figures are committed to the repo under docs/figures, so the
    # site takes them from there rather than depending on a local run having
    # happened — they are the same files the README embeds.
    figures = repo_root / "docs" / "figures"
    if figures.is_dir():
        destination = site / "public" / "diversification"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("absorption_ratio.png", "absorption_regimes.png"):
            source = figures / name
            if source.exists():
                shutil.copy2(source, destination / name)
                print(f"copied {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
