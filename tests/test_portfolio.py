"""Portfolio aggregation tests, including the equity + option integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.data import synthetic_prices, to_returns
from risk_engine.options import black_scholes_price
from risk_engine.portfolio import Portfolio
from risk_engine.var import historical_var


@pytest.fixture
def mixed_book() -> Portfolio:
    """Long equity, long a call on the same name, plus an unrelated holding."""
    book = Portfolio()
    book.add_equity("AAPL", quantity=1000, price=190.0)
    book.add_equity("MSFT", quantity=400, price=410.0)
    book.add_option(
        underlying="AAPL", kind="call", strike=200.0, expiry_years=0.5,
        volatility=0.28, quantity=10, spot=190.0, rate=0.04,
    )
    return book


def test_market_value_sums_all_legs(mixed_book):
    option_value = 10 * 100 * black_scholes_price(190.0, 200.0, 0.5, 0.04, 0.28, "call")
    expected = 1000 * 190.0 + 400 * 410.0 + option_value
    assert mixed_book.market_value == pytest.approx(expected)


def test_underlyings_are_deduplicated_and_ordered(mixed_book):
    assert mixed_book.underlyings == ["AAPL", "MSFT"]


def test_delta_equivalent_exposure_exceeds_cash_equity(mixed_book):
    """A long call adds positive delta, so AAPL exposure must exceed the shares alone."""
    exposure = mixed_book.exposure_by_underlying()
    assert exposure["AAPL"] > 1000 * 190.0
    assert exposure["MSFT"] == pytest.approx(400 * 410.0)


def test_equity_contributes_only_delta(mixed_book):
    greeks = mixed_book.greeks_by_underlying()
    assert greeks.loc["MSFT", "delta"] == pytest.approx(400.0)
    for higher_order in ("gamma", "vega", "theta", "rho"):
        assert greeks.loc["MSFT", higher_order] == pytest.approx(0.0)


def test_option_leg_contributes_positive_gamma_and_vega(mixed_book):
    greeks = mixed_book.greeks_by_underlying()
    assert greeks.loc["AAPL", "gamma"] > 0
    assert greeks.loc["AAPL", "vega"] > 0
    assert greeks.loc["AAPL", "theta"] < 0  # a long option decays


def test_equity_only_pnl_is_exactly_linear():
    """With no options the revaluation must be exact, not approximate."""
    book = Portfolio().add_equity("A", 100, 50.0)
    scenarios = pd.DataFrame({"A": [-0.10, 0.0, 0.05]})
    assert book.scenario_pnl(scenarios) == pytest.approx([-500.0, 0.0, 250.0])


def test_gamma_makes_option_pnl_convex():
    """A long call gains more on an up-move than it loses on an equal down-move.

    A delta-only approximation would make these symmetric; the gamma term is
    what makes the option book's risk profile correct.
    """
    book = Portfolio().add_option(
        underlying="A", kind="call", strike=100.0, expiry_years=1.0,
        volatility=0.25, quantity=1, spot=100.0,
    )
    scenarios = pd.DataFrame({"A": [-0.05, 0.05]})
    down, up = book.scenario_pnl(scenarios)
    assert up > abs(down)


def test_long_put_hedge_reduces_downside():
    """Adding a protective put must shrink the loss in the worst scenario."""
    naked = Portfolio().add_equity("A", 100, 100.0)
    hedged = Portfolio().add_equity("A", 100, 100.0).add_option(
        underlying="A", kind="put", strike=95.0, expiry_years=0.5,
        volatility=0.30, quantity=1, spot=100.0,
    )
    crash = pd.DataFrame({"A": [-0.15]})
    assert hedged.scenario_pnl(crash)[0] > naked.scenario_pnl(crash)[0]


def test_missing_scenario_column_is_an_error(mixed_book):
    with pytest.raises(ValueError, match="missing columns"):
        mixed_book.scenario_pnl(pd.DataFrame({"AAPL": [0.01]}))


def test_end_to_end_var_on_a_mixed_book():
    """The integration test: one VaR number covering equity and options together.

    This is what the four original notebooks could not produce, because the
    option model and the VaR model shared no data path.
    """
    prices = synthetic_prices(["AAPL", "MSFT"], n_days=1500, seed=17)
    returns = to_returns(prices)

    book = Portfolio()
    book.add_equity("AAPL", 500, float(prices["AAPL"].iloc[-1]))
    book.add_equity("MSFT", 300, float(prices["MSFT"].iloc[-1]))
    book.add_option(
        underlying="AAPL", kind="put", strike=float(prices["AAPL"].iloc[-1]) * 0.95,
        expiry_years=0.25, volatility=0.30, quantity=5,
        spot=float(prices["AAPL"].iloc[-1]),
    )

    pnl_returns = book.scenario_returns(returns)
    result = historical_var(pnl_returns, 0.99)

    assert result.var > 0
    assert result.expected_shortfall > result.var
    assert np.isfinite(result.var)
