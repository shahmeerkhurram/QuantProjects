"""Market data loading, with an on-disk cache and a deterministic offline fallback.

The engine must be runnable by someone who clones the repository on a plane, and
by CI, which has no business making network calls. So every loader here degrades
gracefully:

    live download  ->  local CSV cache  ->  synthetic generator

The synthetic generator is not a toy: it produces correlated Student-t returns
with volatility clustering, which is precisely the structure that makes a
normal-assumption VaR fail its backtest. That makes it useful as a *known-truth*
fixture for the test suite, not just a placeholder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["CACHE_DIR", "load_prices", "synthetic_prices", "to_returns"]

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _cache_path(tickers: list[str], start: str, end: str | None) -> Path:
    key = f"{'-'.join(sorted(tickers))}_{start}_{end or 'latest'}".replace("/", "")
    return CACHE_DIR / f"{key}.csv"


def load_prices(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
    use_cache: bool = True,
    allow_synthetic: bool = True,
) -> pd.DataFrame:
    """Load adjusted close prices for ``tickers``.

    Tries the cache first, then yfinance, then the synthetic generator. Returns a
    DataFrame indexed by date with one column per ticker, forward-filled across
    non-overlapping holidays and with any all-NaN column dropped.
    """
    if not tickers:
        raise ValueError("no tickers requested")
    path = _cache_path(tickers, start, end)

    if use_cache and path.exists():
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if not frame.empty:
            return _clean(frame, tickers)

    frame = _download(tickers, start, end)
    if frame is not None and not frame.empty:
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path)
        return _clean(frame, tickers)

    if not allow_synthetic:
        raise RuntimeError(
            "could not download market data and synthetic fallback is disabled"
        )
    return synthetic_prices(tickers, start=start, end=end)


def _download(tickers: list[str], start: str, end: str | None) -> pd.DataFrame | None:
    """Fetch from yfinance, returning ``None`` on any failure.

    yfinance is an optional dependency and an unreliable network endpoint, so
    every failure mode collapses to the same fallback path rather than crashing
    a risk run.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False
        )
    except Exception:  # noqa: BLE001 — deliberate: see below
        # A blind catch is correct here. yfinance is an unofficial scraper and
        # raises whatever its transport, parser or on-disk cache happens to
        # raise — HTTP errors, JSON decode errors, timezone-cache sqlite locks.
        # Enumerating those would be a guess that goes stale on every upgrade,
        # and every one of them means the same thing to this function: no data,
        # fall back. The failure is contained, not swallowed — the caller sees
        # ``None`` and moves to the next source in the chain.
        return None
    if raw is None or raw.empty:
        return None

    # yfinance returns a MultiIndex for multiple tickers and a flat frame for one.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return None
        close = raw["Close"]
    else:
        if "Close" not in raw.columns:
            return None
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return close


def _clean(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    keep = [t for t in tickers if t in frame.columns]
    if keep:
        frame = frame[keep]
    frame = frame.dropna(axis=1, how="all").ffill().dropna()
    if frame.empty:
        raise ValueError("price frame is empty after cleaning")
    return frame


def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Convert prices to simple (default) or log returns.

    Simple returns are the default because they aggregate correctly *across
    assets* — a portfolio's simple return is the weighted sum of its holdings'
    simple returns, which log returns do not satisfy. Log returns are offered for
    the single-asset time-aggregation case where they are the better choice.
    """
    if prices.empty:
        raise ValueError("price frame is empty")
    returns = np.log(prices / prices.shift(1)) if log else prices.pct_change()
    return returns.dropna(how="any")


def synthetic_prices(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
    n_days: int | None = None,
    seed: int = 7,
    df: float = 4.0,
    base_vol: float = 0.015,
    correlation: float = 0.55,
) -> pd.DataFrame:
    """Generate correlated fat-tailed prices with volatility clustering.

    The process is a Student-t copula over a GARCH-like variance recursion. The
    resulting series deliberately violates the normality assumption, so a
    parametric-normal VaR run against it will fail its Kupiec test — which is the
    behaviour the tests assert.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    if n_days is None:
        end_ts = pd.Timestamp(end) if end else pd.Timestamp("2024-12-31")
        n_days = max(len(pd.bdate_range(pd.Timestamp(start), end_ts)), 500)

    corr = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    # Persistent variance: yesterday's shock feeds tomorrow's volatility.
    alpha, beta = 0.08, 0.90
    var_t = np.full(n_assets, base_vol**2)
    long_run = base_vol**2 * (1 - alpha - beta)
    returns = np.empty((n_days, n_assets))

    for t in range(n_days):
        z = rng.standard_normal(n_assets) @ chol.T
        chi = rng.chisquare(df) / df
        shock = z / np.sqrt(chi) * np.sqrt((df - 2.0) / df)
        step = np.sqrt(var_t) * shock
        returns[t] = step
        var_t = long_run + alpha * step**2 + beta * var_t

    index = pd.bdate_range(start=start, periods=n_days, name="Date")
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    return pd.DataFrame(prices, index=index, columns=tickers)
