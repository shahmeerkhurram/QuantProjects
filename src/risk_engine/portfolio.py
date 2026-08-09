"""Portfolio construction across equity and option positions.

This is the module that makes the repository one engine rather than three
unrelated scripts: the option book and the equity book are revalued against the
*same* simulated return scenarios, so a single VaR number covers both.

Options are revalued by a delta-gamma expansion rather than by full repricing on
every path. The trade-off is deliberate and documented in
:meth:`Portfolio.scenario_pnl`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .options import OptionType, black_scholes_greeks

__all__ = ["EquityPosition", "OptionPosition", "Portfolio"]


@dataclass(frozen=True)
class EquityPosition:
    """A cash equity holding. Negative ``quantity`` is a short."""

    ticker: str
    quantity: float
    price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.price


@dataclass(frozen=True)
class OptionPosition:
    """A European option on ``underlying``.

    ``contract_size`` defaults to 100, the US listed-equity convention; set it to
    1 for OTC or index-point quoting.
    """

    underlying: str
    kind: OptionType
    strike: float
    expiry_years: float
    volatility: float
    quantity: float
    spot: float
    rate: float = 0.04
    dividend_yield: float = 0.0
    contract_size: float = 100.0

    def greeks(self):
        return black_scholes_greeks(
            self.spot,
            self.strike,
            self.expiry_years,
            self.rate,
            self.volatility,
            self.kind,
            self.dividend_yield,
        )

    @property
    def market_value(self) -> float:
        return self.quantity * self.contract_size * self.greeks().price


@dataclass
class Portfolio:
    """A book of equity and option positions valued in a single currency."""

    equities: list[EquityPosition] = field(default_factory=list)
    options: list[OptionPosition] = field(default_factory=list)

    def add_equity(self, ticker: str, quantity: float, price: float) -> Portfolio:
        self.equities.append(EquityPosition(ticker, quantity, price))
        return self

    def add_option(self, **kwargs) -> Portfolio:
        self.options.append(OptionPosition(**kwargs))
        return self

    @property
    def underlyings(self) -> list[str]:
        """Every distinct risk factor in the book, in stable order."""
        seen: dict[str, None] = {}
        for pos in self.equities:
            seen[pos.ticker] = None
        for opt in self.options:
            seen[opt.underlying] = None
        return list(seen)

    @property
    def market_value(self) -> float:
        return sum(p.market_value for p in self.equities) + sum(
            o.market_value for o in self.options
        )

    def exposure_by_underlying(self) -> pd.Series:
        """Delta-equivalent notional exposure per risk factor.

        An option contributes ``delta * contract_size * quantity * spot``, which
        is the equity position it behaves like for small moves. This is the
        correct denominator for a concentration limit — option notional is not.
        """
        exposure = {name: 0.0 for name in self.underlyings}
        for pos in self.equities:
            exposure[pos.ticker] += pos.market_value
        for opt in self.options:
            g = opt.greeks()
            exposure[opt.underlying] += g.delta * opt.contract_size * opt.quantity * opt.spot
        return pd.Series(exposure, name="delta_equivalent_exposure")

    def greeks_by_underlying(self) -> pd.DataFrame:
        """Aggregate option Greeks per underlying, scaled by position size.

        Equity positions contribute delta only (one unit of delta per share) and
        zero to every higher-order Greek.
        """
        rows: dict[str, dict[str, float]] = {
            name: {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
            for name in self.underlyings
        }
        for pos in self.equities:
            rows[pos.ticker]["delta"] += pos.quantity
        for opt in self.options:
            g = opt.greeks()
            scale = opt.quantity * opt.contract_size
            rows[opt.underlying]["delta"] += g.delta * scale
            rows[opt.underlying]["gamma"] += g.gamma * scale
            rows[opt.underlying]["vega"] += g.vega * scale
            rows[opt.underlying]["theta"] += g.theta * scale
            rows[opt.underlying]["rho"] += g.rho * scale
        return pd.DataFrame(rows).T.rename_axis("underlying")

    def spot_map(self) -> dict[str, float]:
        """Current spot per risk factor, taken from whichever leg supplies it."""
        spots: dict[str, float] = {}
        for pos in self.equities:
            spots[pos.ticker] = pos.price
        for opt in self.options:
            spots.setdefault(opt.underlying, opt.spot)
        return spots

    def scenario_pnl(self, scenarios: pd.DataFrame) -> np.ndarray:
        """Revalue the whole book across a matrix of underlying returns.

        ``scenarios`` has one column per underlying and one row per scenario.

        Equity legs are linear and exact. Option legs use a second-order
        (delta-gamma) expansion:

            dV ~ delta * dS + 0.5 * gamma * dS^2

        Full repricing on every path would be more accurate but is quadratically
        more expensive, and the gamma term already captures the convexity that
        makes a delta-only VaR wrong for an option book. The approximation
        degrades for very large moves and for options close to expiry — both
        stated here because the limitation is real, not hidden.
        """
        missing = set(self.underlyings) - set(scenarios.columns)
        if missing:
            raise ValueError(f"scenarios are missing columns for: {sorted(missing)}")

        n = len(scenarios)
        pnl = np.zeros(n)
        spots = self.spot_map()

        for pos in self.equities:
            pnl += pos.market_value * scenarios[pos.ticker].to_numpy(dtype=float)

        for opt in self.options:
            ret = scenarios[opt.underlying].to_numpy(dtype=float)
            d_spot = spots[opt.underlying] * ret
            g = opt.greeks()
            scale = opt.quantity * opt.contract_size
            pnl += scale * (g.delta * d_spot + 0.5 * g.gamma * d_spot**2)

        return pnl

    def scenario_returns(self, scenarios: pd.DataFrame) -> np.ndarray:
        """Scenario P&L expressed as a fraction of current market value."""
        mv = self.market_value
        if np.isclose(mv, 0.0):
            raise ValueError("portfolio market value is zero; cannot express returns")
        return self.scenario_pnl(scenarios) / mv

    def summary(self) -> str:
        lines = [
            f"Market value      : {self.market_value:,.2f}",
            f"Equity positions  : {len(self.equities)}",
            f"Option positions  : {len(self.options)}",
            f"Risk factors      : {', '.join(self.underlyings) or 'none'}",
        ]
        return "\n".join(lines)
