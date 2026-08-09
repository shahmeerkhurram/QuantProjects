"""Export engine results as JSON for the portfolio site to consume.

The website used to carry these numbers as hand-typed literals, which meant that
every rerun of the engine silently invalidated the published page. This script
makes the engine the single source of truth: run it, and the site picks up
whatever the engine currently says.

Usage
-----
    python scripts/export_site_data.py --site ../shahmeerkhurram.github.io

Writes ``src/data/risk-engine-results.json`` and copies the charts into
``public/risk-engine/``.
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
    historical_var,
    load_prices,
    monte_carlo_var,
    parametric_var,
    portfolio_returns,
    rank_systemic_assets,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, help="path to the Astro site repo")
    parser.add_argument("--outputs", default="outputs", help="directory holding charts")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    site = Path(args.site).resolve()
    if not (site / "src" / "data").is_dir():
        raise SystemExit(f"error: {site} does not look like the site repo")

    payload = build_payload(repo_root)
    target = site / "src" / "data" / "risk-engine-results.json"
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
