"""European option pricing, Greeks, implied volatility and a binomial cross-check.

Conventions used throughout this module
---------------------------------------
* Rates and volatilities are annualised decimals (``r=0.05`` is 5% per year).
* ``T`` is time to expiry in years.
* Greeks are returned in *raw* mathematical units:
      delta  : dV/dS            (per 1.00 of underlying)
      gamma  : d2V/dS2          (per 1.00 of underlying, squared)
      vega   : dV/dsigma        (per 1.00 = 100 vol points)
      theta  : dV/dT_calendar   (per 1.00 year, already negated so it is decay)
      rho    : dV/dr            (per 1.00 = 100 rate points)
  Desks usually quote vega per 1 vol point and theta per calendar day. Use
  :meth:`Greeks.quoted` for that convention rather than rescaling by hand.
* A continuous dividend yield ``q`` is supported; it defaults to zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]

__all__ = [
    "Greeks",
    "OptionType",
    "binomial_price",
    "black_scholes_greeks",
    "black_scholes_price",
    "implied_volatility",
    "put_call_parity_gap",
]


def _validate(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError(f"spot must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"strike must be positive, got {K}")
    if T < 0:
        raise ValueError(f"time to expiry cannot be negative, got {T}")
    if sigma < 0:
        raise ValueError(f"volatility cannot be negative, got {sigma}")


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float):
    """Return the Black-Scholes ``d1``/``d2`` terms.

    ``sigma * sqrt(T)`` appears in the denominator, so the degenerate cases
    ``T == 0`` (expiry) and ``sigma == 0`` (deterministic forward) are handled by
    the callers, not here.
    """
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _intrinsic(S: float, K: float, T: float, r: float, q: float, kind: OptionType) -> float:
    """Discounted intrinsic value: the correct limit as vol or time goes to zero."""
    forward = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)
    if kind == "call":
        return max(forward - discounted_strike, 0.0)
    return max(discounted_strike - forward, 0.0)


@dataclass(frozen=True)
class Greeks:
    """First- and second-order sensitivities in raw units (see module docstring)."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def quoted(self) -> dict[str, float]:
        """Rescale to the units a trading desk actually quotes.

        vega per 1 volatility point, theta per calendar day, rho per 1 basis
        point move in rates.
        """
        return {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega_per_vol_point": self.vega / 100.0,
            "theta_per_day": self.theta / 365.0,
            "rho_per_bp": self.rho / 10_000.0,
        }

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: OptionType = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes-Merton price of a European option.

    Falls back to discounted intrinsic value when ``T`` or ``sigma`` is zero,
    which keeps the function total instead of dividing by zero.
    """
    _validate(S, K, T, sigma)
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if T == 0.0 or sigma == 0.0:
        return _intrinsic(S, K, T, r, q, kind)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_spot = S * np.exp(-q * T)
    disc_strike = K * np.exp(-r * T)
    if kind == "call":
        return float(disc_spot * norm.cdf(d1) - disc_strike * norm.cdf(d2))
    return float(disc_strike * norm.cdf(-d2) - disc_spot * norm.cdf(-d1))


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: OptionType = "call",
    q: float = 0.0,
) -> Greeks:
    """All five standard Greeks alongside the price."""
    _validate(S, K, T, sigma)
    price = black_scholes_price(S, K, T, r, sigma, kind, q)
    if T == 0.0 or sigma == 0.0:
        # At the boundary the option is either dead or a pure forward; the only
        # non-zero sensitivity is delta, and only when it finishes in the money.
        itm = (kind == "call" and S > K) or (kind == "put" and S < K)
        sign = 1.0 if kind == "call" else -1.0
        return Greeks(price, sign * float(itm), 0.0, 0.0, 0.0, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t
    # Common decay term shared by calls and puts.
    decay = -(S * disc_q * pdf_d1 * sigma) / (2 * sqrt_t)

    if kind == "call":
        delta = disc_q * norm.cdf(d1)
        theta = decay - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = decay + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2)

    return Greeks(
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def put_call_parity_gap(
    call: float, put: float, S: float, K: float, T: float, r: float, q: float = 0.0
) -> float:
    """Residual of ``C - P - (S e^{-qT} - K e^{-rT})``.

    Zero for arbitrage-free European prices. Used as a correctness assertion in
    the test suite rather than as a trading signal.
    """
    return float(call - put - (S * np.exp(-q * T) - K * np.exp(-r * T)))


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    kind: OptionType = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_vol: float = 5.0,
) -> float:
    """Invert Black-Scholes for volatility via Brent's method.

    Brent is used instead of Newton because it cannot diverge: the price is
    strictly increasing in ``sigma``, so bracketing is guaranteed once the quote
    sits inside the no-arbitrage bounds.

    Raises
    ------
    ValueError
        If the quoted price violates the no-arbitrage bounds, or if the implied
        volatility exceeds ``max_vol``.
    """
    _validate(S, K, T, 0.0)
    if T == 0.0:
        raise ValueError("implied volatility is undefined at expiry (T=0)")

    lower = _intrinsic(S, K, T, r, q, kind)
    upper = S * np.exp(-q * T) if kind == "call" else K * np.exp(-r * T)
    if not (lower - tol <= price <= upper + tol):
        raise ValueError(
            f"price {price:.6f} is outside the no-arbitrage band "
            f"[{lower:.6f}, {upper:.6f}] for a {kind}"
        )

    def objective(vol: float) -> float:
        return black_scholes_price(S, K, T, r, vol, kind, q) - price

    if objective(max_vol) < 0:
        raise ValueError(f"implied volatility exceeds max_vol={max_vol}")
    # A vanishing vol gives the intrinsic value, so the bracket is valid by
    # construction once the band check above has passed.
    return float(brentq(objective, 1e-12, max_vol, xtol=tol, maxiter=200))


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: OptionType = "call",
    q: float = 0.0,
    steps: int = 500,
    american: bool = False,
) -> float:
    """Cox-Ross-Rubinstein lattice price.

    Independent of the closed-form implementation, so the European case is used
    in the tests as a convergence check on :func:`black_scholes_price`. The
    American case exists because early exercise has no closed form.
    """
    _validate(S, K, T, sigma)
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if T == 0.0 or sigma == 0.0:
        return _intrinsic(S, K, T, r, q, kind)

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability {p:.4f} left (0,1); increase steps or check inputs"
        )
    discount = np.exp(-r * dt)

    # Terminal spot ladder, then roll the lattice backwards in place.
    j = np.arange(steps + 1)
    spot = S * u**j * d ** (steps - j)
    values = np.maximum(spot - K, 0.0) if kind == "call" else np.maximum(K - spot, 0.0)

    for step in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            j = np.arange(step + 1)
            spot = S * u**j * d ** (step - j)
            exercise = spot - K if kind == "call" else K - spot
            values = np.maximum(values, exercise)

    return float(values[0])
