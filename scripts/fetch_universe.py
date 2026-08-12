"""One-off: pull the diversification study's universe into the on-disk cache."""

from __future__ import annotations

import sys

from risk_engine.data import load_prices

UNIVERSE = [
    "AAPL", "MSFT", "INTC", "IBM", "ORCL", "TXN",          # technology
    "JPM", "BAC", "GS", "AXP",                              # financials
    "JNJ", "PFE", "MRK", "UNH",                             # health care
    "XOM", "CVX", "SLB",                                    # energy
    "PG", "KO", "WMT", "MCD",                               # staples / discretionary
    "CAT", "HON", "UNP",                                    # industrials
    "NEE", "SO",                                            # utilities
]

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2007-01-01"
    prices = load_prices(UNIVERSE, start=start, allow_synthetic=False)
    print(prices.shape, prices.index[0].date(), prices.index[-1].date())
    print(sorted(prices.columns))
