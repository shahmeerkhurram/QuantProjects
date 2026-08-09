"""Option pricing tests.

These assert *financial* properties — no-arbitrage relationships, convergence
against an independent numerical method, analytic limits — rather than merely
checking that the code returns a number.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from risk_engine.options import (
    binomial_price,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
    put_call_parity_gap,
)

CASES = [
    # S,    K,    T,    r,     sigma, q
    (100.0, 100.0, 1.00, 0.05, 0.20, 0.00),  # at the money
    (100.0, 130.0, 0.50, 0.03, 0.35, 0.00),  # far out of the money
    (100.0, 70.0, 2.00, 0.05, 0.15, 0.02),   # deep in the money, with dividend
    (50.0, 55.0, 0.25, 0.01, 0.60, 0.00),    # short dated, high vol
    (250.0, 240.0, 3.00, 0.06, 0.25, 0.015),
]


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_put_call_parity_holds(S, K, T, r, sigma, q):
    """C - P = S*e^{-qT} - K*e^{-rT} must hold exactly for European options.

    This is the single strongest correctness check available for a pricer: it is
    model-free, so an error in either the call or the put branch breaks it.
    """
    call = black_scholes_price(S, K, T, r, sigma, "call", q)
    put = black_scholes_price(S, K, T, r, sigma, "put", q)
    assert abs(put_call_parity_gap(call, put, S, K, T, r, q)) < 1e-9


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_binomial_converges_to_black_scholes(S, K, T, r, sigma, q, kind):
    """A CRR lattice must converge to the closed form as steps increase.

    The lattice shares no code with the analytic formula, so agreement is real
    evidence rather than a tautology.
    """
    analytic = black_scholes_price(S, K, T, r, sigma, kind, q)
    lattice = binomial_price(S, K, T, r, sigma, kind, q, steps=2000)
    assert abs(lattice - analytic) < 0.01 * max(1.0, analytic)


def test_binomial_error_shrinks_with_steps():
    """Convergence should be monotone in the step count, not accidental."""
    args = (100.0, 105.0, 1.0, 0.05, 0.2, "call", 0.0)
    analytic = black_scholes_price(*args[:5], args[5], args[6])
    coarse = abs(binomial_price(*args, steps=25) - analytic)
    fine = abs(binomial_price(*args, steps=2000) - analytic)
    assert fine < coarse


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trips(S, K, T, r, sigma, q, kind):
    """Pricing at sigma then inverting must recover sigma."""
    price = black_scholes_price(S, K, T, r, sigma, kind, q)
    assert implied_volatility(price, S, K, T, r, kind, q) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_arbitrage_violating_price():
    """A price below intrinsic value has no implied volatility; say so loudly."""
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(0.001, 100.0, 50.0, 1.0, 0.05, "call")


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_delta_matches_numerical_derivative(S, K, T, r, sigma, q):
    """Analytic delta must equal a central finite difference of the price."""
    h = S * 1e-5
    up = black_scholes_price(S + h, K, T, r, sigma, "call", q)
    down = black_scholes_price(S - h, K, T, r, sigma, "call", q)
    numerical = (up - down) / (2 * h)
    assert black_scholes_greeks(S, K, T, r, sigma, "call", q).delta == pytest.approx(
        numerical, rel=1e-4
    )


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_gamma_matches_second_derivative(S, K, T, r, sigma, q):
    """Gamma must equal the second finite difference of the price."""
    h = S * 1e-3
    up = black_scholes_price(S + h, K, T, r, sigma, "call", q)
    mid = black_scholes_price(S, K, T, r, sigma, "call", q)
    down = black_scholes_price(S - h, K, T, r, sigma, "call", q)
    numerical = (up - 2 * mid + down) / h**2
    assert black_scholes_greeks(S, K, T, r, sigma, "call", q).gamma == pytest.approx(
        numerical, rel=1e-3
    )


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_vega_matches_numerical_derivative(S, K, T, r, sigma, q):
    h = 1e-6
    up = black_scholes_price(S, K, T, r, sigma + h, "call", q)
    down = black_scholes_price(S, K, T, r, sigma - h, "call", q)
    assert black_scholes_greeks(S, K, T, r, sigma, "call", q).vega == pytest.approx(
        (up - down) / (2 * h), rel=1e-4
    )


def test_gamma_and_vega_are_identical_for_call_and_put():
    """Parity implies the two second-order Greeks cannot differ by option type."""
    call = black_scholes_greeks(100, 105, 1.0, 0.05, 0.2, "call")
    put = black_scholes_greeks(100, 105, 1.0, 0.05, 0.2, "put")
    assert call.gamma == pytest.approx(put.gamma)
    assert call.vega == pytest.approx(put.vega)
    # Delta parity: delta_call - delta_put = e^{-qT} = 1 when q = 0.
    assert call.delta - put.delta == pytest.approx(1.0)


def test_price_is_monotone_in_volatility():
    """Vega is positive, so price must strictly increase with volatility."""
    prices = [black_scholes_price(100, 105, 1.0, 0.05, v) for v in np.linspace(0.05, 1.0, 40)]
    assert all(b > a for a, b in itertools.pairwise(prices))


def test_zero_volatility_gives_discounted_intrinsic():
    """With no uncertainty the option is a forward contract."""
    price = black_scholes_price(100, 90, 1.0, 0.05, 0.0, "call")
    assert price == pytest.approx(100 - 90 * np.exp(-0.05))


def test_expiry_gives_intrinsic_and_no_greeks():
    g = black_scholes_greeks(110, 100, 0.0, 0.05, 0.2, "call")
    assert g.price == pytest.approx(10.0)
    assert g.delta == pytest.approx(1.0)
    assert (g.gamma, g.vega, g.theta, g.rho) == (0.0, 0.0, 0.0, 0.0)


def test_american_put_is_worth_at_least_the_european():
    """Early exercise is an extra right; it cannot reduce value."""
    euro = binomial_price(100, 120, 1.0, 0.08, 0.3, "put", steps=800, american=False)
    amer = binomial_price(100, 120, 1.0, 0.08, 0.3, "put", steps=800, american=True)
    assert amer >= euro - 1e-9
    assert amer > euro  # with these inputs the premium is strictly positive


def test_american_call_without_dividends_equals_european():
    """Merton's result: never optimal to exercise an American call early when q=0."""
    euro = binomial_price(100, 95, 1.0, 0.05, 0.25, "call", 0.0, steps=800, american=False)
    amer = binomial_price(100, 95, 1.0, 0.05, 0.25, "call", 0.0, steps=800, american=True)
    assert amer == pytest.approx(euro, rel=1e-6)


@pytest.mark.parametrize("bad", [{"S": -1}, {"K": 0}, {"T": -0.5}, {"sigma": -0.2}])
def test_invalid_inputs_raise(bad):
    kwargs = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2} | bad
    with pytest.raises(ValueError):
        black_scholes_price(**kwargs)
