"""Expected Shortfall backtesting — Acerbi-Székely on the published VaR universe.

Same tickers, dates and window as the VaR backtesting results already published,
so the ES verdicts can be read directly against the coverage-test verdicts
rather than against a differently-specified run.

    python scripts/es_backtest_study.py --output outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from risk_engine import load_prices, portfolio_returns, to_returns
from risk_engine.backtest import METHODS, compare_es_models, rolling_var_backtest

# The universe the VaR posts and the README already use. Not the 26-name
# diversification universe: the point is comparability with what is published.
TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
CONFIDENCE = 0.975   # the level Basel FRTB moved Expected Shortfall to
WINDOW = 250
N_SIMULATIONS = 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    args = parser.parse_args()

    returns = to_returns(load_prices(TICKERS, start=args.start))
    pnl = portfolio_returns(returns)

    print("=" * 78)
    print("UNIVERSE")
    print(f"   {', '.join(TICKERS)}")
    print(f"   {pnl.index[0].date()} to {pnl.index[-1].date()}  ({len(pnl)} observations)")
    print(f"   confidence {CONFIDENCE:.1%}   window {WINDOW}d   "
          f"models {len(METHODS)}   simulations {args.simulations}")

    table = compare_es_models(
        pnl, confidence=CONFIDENCE, window=WINDOW,
        n_simulations=args.simulations,
    )

    print()
    print("=" * 78)
    print("ES BACKTESTS (Acerbi-Székely) ALONGSIDE THE VaR COVERAGE TESTS")
    shown = table.copy()
    for col in ("breach_rate", "mean_var", "mean_es"):
        shown[col] = shown[col].map("{:.4%}".format)
    for col in ("z1", "z2"):
        shown[col] = shown[col].map("{:+.4f}".format)
    for col in ("z1_p", "z2_p", "kupiec_p", "christoffersen_p"):
        shown[col] = shown[col].map("{:.4f}".format)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(shown.to_string(index=False))

    # The sentence the post is looking for: a model that passes one family of
    # tests and fails the other.
    print()
    print("=" * 78)
    print("AGREEMENT BETWEEN THE VaR AND ES VERDICTS")
    disagree = table[table["var_passes_all"] != table["z2_pass"]]
    if disagree.empty:
        print("   Every model gets the same verdict from both families.")
    else:
        for _, row in disagree.iterrows():
            var_verdict = "passes" if row["var_passes_all"] else "fails"
            es_verdict = "passes" if row["z2_pass"] else "fails"
            print(f"   {row['method']:<20} VaR coverage {var_verdict}, "
                  f"ES (Z2) {es_verdict}   Z2={row['z2']:+.4f} p={row['z2_p']:.4f}")

    print()
    print(f"   VaR coverage passed by : {int(table['var_passes_all'].sum())} of {len(table)}")
    print(f"   ES Z2 passed by        : {int(table['z2_pass'].sum())} of {len(table)}")
    print(f"   ES Z1 passed by        : {int(table['z1_pass'].sum())} of {len(table)}")

    print()
    print("=" * 78)
    print("PER-MODEL DETAIL")
    for method in METHODS:
        bt = rolling_var_backtest(pnl, CONFIDENCE, WINDOW, method)
        from risk_engine.backtest import es_backtest

        z1 = es_backtest(bt, n_simulations=args.simulations, test="z1")
        z2 = es_backtest(bt, n_simulations=args.simulations, test="z2")
        assert bt.es_forecasts is not None
        print(f"\n{method}")
        print(f"   breaches {bt.n_breaches}/{bt.n_observations} "
              f"({bt.breach_rate:.2%})   mean VaR {bt.var_forecasts.mean():.4%}   "
              f"mean ES {bt.es_forecasts.mean():.4%}   "
              f"ES/VaR {float((bt.es_forecasts / bt.var_forecasts).mean()):.4f}")
        print(f"   {z1}")
        print(f"   {z2}")
        print(f"   -> {z2.interpretation}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "es_backtest_comparison.csv", index=False)
    summary = {
        "universe": TICKERS,
        "start": str(pnl.index[0].date()),
        "end": str(pnl.index[-1].date()),
        "observations": len(pnl),
        "confidence": CONFIDENCE,
        "window": WINDOW,
        "simulations": args.simulations,
        "models": list(METHODS),
        "results": table.to_dict("records"),
    }
    (out / "es_backtest_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out / 'es_backtest_comparison.csv'} and "
          f"{out / 'es_backtest_summary.json'}")


if __name__ == "__main__":
    main()
