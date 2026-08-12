"""Command-line interface: ``risk-engine <command>``.

Five commands, mirroring the five questions the engine answers:

    risk-engine report    --tickers AAPL,MSFT,GOOGL --confidence 0.99
    risk-engine backtest  --tickers AAPL,MSFT,GOOGL --window 250
    risk-engine option    --spot 100 --strike 105 --expiry 1 --vol 0.2
    risk-engine contagion --tickers AAPL,MSFT,GOOGL,JPM,XOM
    risk-engine diversification --tickers <20+ names> --benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .backtest import METHODS, compare_models, rolling_var_backtest
from .contagion import correlation_network, rank_systemic_assets
from .data import load_prices, to_returns
from .diversification import (
    drawdown_events,
    event_study,
    realised_vol_signal,
    rolling_absorption_ratio,
    standardised_shift,
    threshold_crossings,
)
from .options import (
    OptionType,
    binomial_price,
    black_scholes_greeks,
    implied_volatility,
    put_call_parity_gap,
)
from .report import (
    plot_absorption_ratio,
    plot_greeks_profile,
    plot_network,
    plot_return_distribution,
    plot_var_backtest,
    results_table,
    save_figure,
)
from .var import (
    conditional_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
    portfolio_returns,
)

DEFAULT_TICKERS = "AAPL,MSFT,GOOGL,JPM,XOM"


def _tickers(raw: str) -> list[str]:
    names = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not names:
        raise argparse.ArgumentTypeError("no tickers supplied")
    return names


def _weights(raw: str | None, n: int):
    if not raw:
        return None
    parts = [float(x) for x in raw.split(",") if x.strip()]
    if len(parts) != n:
        raise SystemExit(f"error: got {len(parts)} weights for {n} tickers")
    return parts


def _rule(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


def _load(args) -> pd.DataFrame:
    prices = load_prices(args.tickers, start=args.start, end=args.end)
    returns = to_returns(prices)
    if returns.shape[0] < 60:
        raise SystemExit("error: fewer than 60 return observations; widen the date range")
    return returns


def cmd_report(args) -> int:
    returns = _load(args)
    weights = _weights(args.weights, returns.shape[1])
    pnl = portfolio_returns(returns, weights)

    results = [
        historical_var(pnl, args.confidence, args.horizon),
        parametric_var(pnl, args.confidence, args.horizon, "normal"),
        parametric_var(pnl, args.confidence, args.horizon, "student_t"),
        monte_carlo_var(returns, weights, args.confidence, args.horizon, "gaussian",
                        seed=args.seed),
        monte_carlo_var(returns, weights, args.confidence, args.horizon, "student_t",
                        seed=args.seed),
        monte_carlo_var(returns, weights, args.confidence, args.horizon, "bootstrap",
                        seed=args.seed),
        conditional_var(pnl, args.confidence, args.horizon, model="ewma",
                        innovation="empirical"),
        conditional_var(pnl, args.confidence, args.horizon, model="garch",
                        innovation="empirical"),
    ]

    print(_rule(f"Portfolio risk report — {', '.join(returns.columns)}"))
    print(f"Observations : {len(pnl)}  ({pnl.index[0].date()} to {pnl.index[-1].date()})")
    print(f"Confidence   : {args.confidence:.1%}   horizon: {args.horizon}d")

    table = results_table(results)
    display = table[["method", "VaR", "ES", "detail"]].copy()
    display["VaR"] = display["VaR"].map("{:.4%}".format)
    display["ES"] = display["ES"].map("{:.4%}".format)
    print(_rule("VaR and Expected Shortfall"))
    print(display.to_string(index=False))

    spread = table["VaR"].max() - table["VaR"].min()
    print(
        f"\nSpread across methods: {spread:.4%} "
        f"({spread / table['VaR'].min():.1%} of the smallest estimate). "
        "A wide spread is model risk, and it is the reason more than one method is run."
    )

    if args.output:
        out = Path(args.output)
        fig = plot_return_distribution(pnl, results)
        path = save_figure(fig, out / "loss_distribution.png")
        table.to_csv(out / "var_results.csv", index=False)
        print(f"\nWrote {path} and {out / 'var_results.csv'}")
    return 0


def cmd_backtest(args) -> int:
    returns = _load(args)
    weights = _weights(args.weights, returns.shape[1])
    pnl = portfolio_returns(returns, weights)

    methods = METHODS if args.all_models else (
        "historical", "parametric_normal", "parametric_t", "ewma_fhs"
    )

    exit_code = 0
    for method in methods:
        result = rolling_var_backtest(pnl, args.confidence, args.window, method,
                                      args.refit_every)
        print(_rule(f"Backtest — {method}"))
        print(result.summary())
        if any(t.reject_at_5pct for t in result.tests):
            exit_code = max(exit_code, 1)
        if args.output:
            fig = plot_var_backtest(result)
            save_figure(fig, Path(args.output) / f"backtest_{method}.png")

    print(_rule("Model comparison (sorted by joint conditional-coverage p-value)"))
    comparison = compare_models(pnl, args.confidence, args.window, methods,
                                args.refit_every)
    display = comparison.copy()
    display["breach_rate"] = display["breach_rate"].map("{:.2%}".format)
    display["mean_var"] = display["mean_var"].map("{:.3%}".format)
    for col in ("kupiec_p", "christoffersen_p", "joint_p"):
        display[col] = display[col].map("{:.4f}".format)
    print(display.drop(columns=["expected_rate"]).to_string(index=False))

    winners = comparison[comparison["passes_all"]]
    if len(winners):
        best = winners.iloc[0]
        print(
            f"\n{best['method']} passes all three coverage tests "
            f"(joint p = {best['joint_p']:.4f}) — the only models that do are the "
            "volatility-filtered ones."
        )
    else:
        print(
            f"\nNo model passes all three tests at {args.confidence:.1%}. "
            "Compare the Christoffersen column: volatility filtering fixes the "
            "clustering even where coverage remains too thin."
        )

    if args.output:
        comparison.to_csv(Path(args.output) / "model_comparison.csv", index=False)
        print(f"\nWrote backtest charts and comparison table to {args.output}")
    return 0 if not args.strict else exit_code


def cmd_option(args) -> int:
    # argparse `choices` already restricts this to "call"/"put"; the annotation
    # carries that guarantee into the type checker.
    kind: OptionType = args.kind
    g = black_scholes_greeks(
        args.spot, args.strike, args.expiry, args.rate, args.vol, kind, args.dividend
    )
    other: OptionType = "put" if kind == "call" else "call"
    g_other = black_scholes_greeks(
        args.spot, args.strike, args.expiry, args.rate, args.vol, other, args.dividend
    )
    call_price = g.price if kind == "call" else g_other.price
    put_price = g_other.price if kind == "call" else g.price

    print(_rule(f"European {kind} — S={args.spot} K={args.strike} T={args.expiry}y"))
    print(f"Price            : {g.price:.6f}")
    print(f"Delta            : {g.delta:.6f}")
    print(f"Gamma            : {g.gamma:.6f}")
    print(f"Vega  (per vol pt): {g.vega / 100:.6f}")
    print(f"Theta (per day)  : {g.theta / 365:.6f}")
    print(f"Rho   (per bp)   : {g.rho / 10_000:.8f}")

    lattice = binomial_price(
        args.spot, args.strike, args.expiry, args.rate, args.vol, kind,
        args.dividend, steps=args.steps
    )
    gap = put_call_parity_gap(
        call_price, put_price, args.spot, args.strike, args.expiry, args.rate, args.dividend
    )
    print(_rule("Independent validation"))
    print(f"Binomial ({args.steps} steps) : {lattice:.6f}   "
          f"|difference| = {abs(lattice - g.price):.2e}")
    print(f"Put-call parity residual  : {gap:.2e}")

    if args.market_price is not None:
        iv = implied_volatility(
            args.market_price, args.spot, args.strike, args.expiry, args.rate,
            args.kind, args.dividend
        )
        print(f"Implied volatility        : {iv:.4%} (from market price "
              f"{args.market_price})")

    if args.output:
        fig = plot_greeks_profile(args.strike, args.expiry, args.rate, args.vol, kind)
        path = save_figure(fig, Path(args.output) / "greeks_profile.png")
        print(f"\nWrote {path}")
    return 0


def cmd_contagion(args) -> int:
    returns = _load(args)
    graph = correlation_network(returns, method=args.network, threshold=args.threshold)
    if graph.number_of_edges() == 0:
        raise SystemExit(
            f"error: no edges survived threshold={args.threshold}; lower it or use --network mst"
        )

    systemic = rank_systemic_assets(graph, initial_distress=args.shock)
    print(_rule(f"Correlation network — {args.network}"))
    nodes = graph.number_of_nodes()
    possible_edges = max(1, nodes * (nodes - 1) / 2)
    print(f"Nodes: {nodes}   Edges: {graph.number_of_edges()}   "
          f"Density: {graph.number_of_edges() / possible_edges:.2f}")

    print(_rule(f"Systemic ranking under a {args.shock:.0%} idiosyncratic shock"))
    shown = systemic.copy()
    for col in ("DebtRank", "total_loss"):
        shown[col] = shown[col].map("{:.4f}".format)
    shown["amplification"] = shown["amplification"].map("{:.2f}x".format)
    print(shown.to_string(index=False))

    top = systemic.iloc[0]
    print(
        f"\n{top['asset']} is the most systemically important position: shocking it alone "
        f"destroys {top['total_loss']:.2%} of portfolio value, {top['amplification']:.2f}x "
        "the direct hit. Concentration limits belong here first."
    )

    if args.output:
        out = Path(args.output)
        save_figure(plot_network(graph, systemic), out / "contagion_network.png")
        systemic.to_csv(out / "systemic_ranking.csv", index=False)
        print(f"\nWrote {out / 'contagion_network.png'} and {out / 'systemic_ranking.csv'}")
    return 0


def cmd_diversification(args) -> int:
    returns = _load(args)
    if returns.shape[1] < 5:
        raise SystemExit(
            f"error: {returns.shape[1]} assets is too few for an absorption ratio; "
            "K = floor(N/5) degenerates below 5 names — pass 20 or more"
        )
    if returns.shape[0] <= args.window:
        raise SystemExit(
            f"error: {returns.shape[0]} observations for a {args.window}-day window; "
            "widen the date range or shorten --window"
        )

    result = rolling_absorption_ratio(
        returns, window=args.window, k=args.k, cov_model=args.cov_model
    )
    pnl = portfolio_returns(returns, _weights(args.weights, returns.shape[1]))
    signal = standardised_shift(result.absorption, short=args.short, long=args.long)
    crossings = threshold_crossings(signal, threshold=args.threshold)
    events = drawdown_events(pnl, min_depth=args.min_depth)

    print(_rule(f"Absorption ratio — {args.cov_model} covariance"))
    print(f"Assets       : {result.n_assets}   K: {result.k}   window: {result.window}d")
    print(f"Absorption   : median {result.absorption.median():.2%}   "
          f"calm {result.absorption.quantile(0.25):.2%}   "
          f"peak {result.absorption.max():.2%} on {result.absorption.idxmax().date()}")
    bets = result.effective_bets
    print(f"Independent bets: median {bets.median():.2f} of {result.n_assets} holdings   "
          f"stress {bets.quantile(0.05):.2f}   min {bets.min():.2f}")

    print(_rule(f"Drawdown events (peak-to-trough >= {args.min_depth:.0%})"))
    for event in events:
        print(f"  {event}")

    # Only events the signal could have reached are scored; see event_study.
    available_from = signal.first_valid_index()
    study = event_study(crossings, events, returns.index, horizon=args.horizon,
                        available_from=available_from)
    print(_rule(f"Event study — {args.threshold:g} st.dev. crossings"))
    print(study.summary())

    if args.benchmark:
        vol_signal = realised_vol_signal(pnl, short=args.short, long=args.long)
        vol_study = event_study(
            threshold_crossings(vol_signal, threshold=args.threshold), events,
            returns.index, horizon=args.horizon, available_from=available_from,
        )
        print(_rule("Benchmark — trailing realised volatility, identical machinery"))
        print(vol_study.summary())
        # State the comparison rather than leaving the reader to infer it: the
        # benchmark exists to be able to falsify the absorption signal.
        if vol_study.median_lead > study.median_lead or study.n_hits == 0:
            print("\nRealised volatility warns at least as early here. Absorption is "
                  "measuring concentration, not forecasting it.")

    if args.output:
        out = Path(args.output)
        save_figure(
            plot_absorption_ratio(result, events=events, signal=signal,
                                  crossings=crossings, threshold=args.threshold),
            out / "absorption_ratio.png",
        )
        frame = pd.DataFrame(
            {"absorption": result.absorption, "effective_bets": result.effective_bets}
        )
        frame.to_csv(out / "absorption_ratio.csv")
        print(f"\nWrote {out / 'absorption_ratio.png'} and {out / 'absorption_ratio.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risk-engine",
        description="Portfolio risk engine: VaR/ES, regulatory backtesting, "
                    "option Greeks and network contagion.",
    )
    parser.add_argument("--version", action="version", version=f"risk-engine {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_market_args(p):
        p.add_argument("--tickers", type=_tickers, default=_tickers(DEFAULT_TICKERS),
                       help="comma-separated tickers")
        p.add_argument("--start", default="2018-01-01")
        p.add_argument("--end", default=None)
        p.add_argument("--weights", default=None,
                       help="comma-separated portfolio weights (default: equal)")
        p.add_argument("--output", default=None, help="directory for charts and CSVs")

    p_report = sub.add_parser("report", help="VaR and Expected Shortfall across all methods")
    add_market_args(p_report)
    p_report.add_argument("--confidence", type=float, default=0.99)
    p_report.add_argument("--horizon", type=int, default=1, help="holding period in days")
    p_report.add_argument("--seed", type=int, default=42)
    p_report.set_defaults(func=cmd_report)

    p_bt = sub.add_parser("backtest", help="walk-forward backtest with coverage tests")
    add_market_args(p_bt)
    p_bt.add_argument("--confidence", type=float, default=0.99)
    p_bt.add_argument("--window", type=int, default=250)
    p_bt.add_argument("--all-models", action="store_true",
                      help="include every model, incl. GARCH (slower)")
    p_bt.add_argument("--refit-every", type=int, default=25,
                      help="GARCH refit interval during the walk-forward")
    p_bt.add_argument("--strict", action="store_true",
                      help="exit non-zero if any model fails a coverage test")
    p_bt.set_defaults(func=cmd_backtest)

    p_opt = sub.add_parser("option", help="price a European option and show its Greeks")
    p_opt.add_argument("--spot", type=float, required=True)
    p_opt.add_argument("--strike", type=float, required=True)
    p_opt.add_argument("--expiry", type=float, required=True, help="years to expiry")
    p_opt.add_argument("--vol", type=float, required=True, help="annualised volatility")
    p_opt.add_argument("--rate", type=float, default=0.04)
    p_opt.add_argument("--dividend", type=float, default=0.0)
    p_opt.add_argument("--kind", choices=["call", "put"], default="call")
    p_opt.add_argument("--steps", type=int, default=1000, help="binomial lattice steps")
    p_opt.add_argument("--market-price", type=float, default=None,
                       help="if given, solve for implied volatility")
    p_opt.add_argument("--output", default=None)
    p_opt.set_defaults(func=cmd_option)

    p_con = sub.add_parser("contagion", help="correlation network and DebtRank ranking")
    add_market_args(p_con)
    p_con.add_argument("--network", choices=["threshold", "mst", "complete"],
                       default="threshold")
    p_con.add_argument("--threshold", type=float, default=0.3)
    p_con.add_argument("--shock", type=float, default=0.3,
                       help="initial distress applied to the shocked asset")
    p_con.set_defaults(func=cmd_contagion)

    p_div = sub.add_parser(
        "diversification", help="absorption ratio, effective bets and the event study"
    )
    add_market_args(p_div)
    p_div.add_argument("--window", type=int, default=500)
    p_div.add_argument("--k", type=int, default=None,
                       help="retained eigenvectors (default: floor(N/5))")
    p_div.add_argument("--cov-model", choices=["ewma", "sample"], default="ewma")
    p_div.add_argument("--threshold", type=float, default=1.0,
                       help="signal threshold in standard deviations")
    p_div.add_argument("--short", type=int, default=15)
    p_div.add_argument("--long", type=int, default=250)
    p_div.add_argument("--min-depth", type=float, default=0.15,
                       help="peak-to-trough decline defining a drawdown event")
    p_div.add_argument("--horizon", type=int, default=60,
                       help="trading days a warning may precede an onset by")
    p_div.add_argument("--benchmark", action="store_true",
                       help="also score trailing realised volatility, for comparison")
    p_div.set_defaults(func=cmd_diversification)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
