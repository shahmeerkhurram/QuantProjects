"""Portfolio risk engine.

A single toolkit covering the three questions a market-risk function has to
answer, all driven from the same data and the same portfolio object:

* **How much can we lose?**  :mod:`risk_engine.var` — VaR and Expected Shortfall
  by historical, parametric and Monte Carlo methods.
* **Is the model any good?**  :mod:`risk_engine.backtest` — walk-forward
  backtesting with the Kupiec, Christoffersen and joint coverage tests.
* **What happens to the derivatives book, and where does stress spread?**
  :mod:`risk_engine.options`, :mod:`risk_engine.portfolio` and
  :mod:`risk_engine.contagion`.
"""

from .backtest import (
    METHODS,
    BacktestResult,
    CoverageTest,
    basel_traffic_light,
    basel_zone_thresholds,
    christoffersen_independence,
    compare_models,
    conditional_coverage,
    kupiec_pof,
    rolling_var_backtest,
)
from .contagion import (
    ContagionResult,
    correlation_network,
    debt_rank,
    rank_systemic_assets,
)
from .data import load_prices, synthetic_prices, to_returns
from .options import (
    Greeks,
    binomial_price,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
    put_call_parity_gap,
)
from .portfolio import EquityPosition, OptionPosition, Portfolio
from .var import (
    RiskResult,
    conditional_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
    portfolio_returns,
    scale_horizon,
)
from .volatility import (
    GarchParams,
    ewma_forecast,
    ewma_variance,
    fit_garch11,
    garch_forecast,
    garch_variance,
    standardised_residuals,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # data
    "load_prices",
    "to_returns",
    "synthetic_prices",
    # var
    "RiskResult",
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "conditional_var",
    "portfolio_returns",
    "scale_horizon",
    # volatility
    "GarchParams",
    "ewma_variance",
    "ewma_forecast",
    "fit_garch11",
    "garch_variance",
    "garch_forecast",
    "standardised_residuals",
    # backtest
    "METHODS",
    "BacktestResult",
    "CoverageTest",
    "rolling_var_backtest",
    "compare_models",
    "kupiec_pof",
    "christoffersen_independence",
    "conditional_coverage",
    "basel_traffic_light",
    "basel_zone_thresholds",
    # options
    "Greeks",
    "black_scholes_price",
    "black_scholes_greeks",
    "implied_volatility",
    "binomial_price",
    "put_call_parity_gap",
    # portfolio
    "Portfolio",
    "EquityPosition",
    "OptionPosition",
    # contagion
    "ContagionResult",
    "correlation_network",
    "debt_rank",
    "rank_systemic_assets",
]
