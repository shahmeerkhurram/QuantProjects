"""Is the most connected asset the most systemically important?

The README makes that claim on a seven-node network. This quantifies it on the
same 26-name, seven-sector universe as the diversification study, so the two
write-ups describe the same portfolio, and checks whether the answer survives
reasonable changes to how the network is built.

    python scripts/contagion_study.py --output outputs
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy import stats

from risk_engine import (
    correlation_network,
    debt_rank,
    load_prices,
    rank_systemic_assets,
    to_returns,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diversification_study import UNIVERSE

# Network construction, stated once and applied everywhere. A ranking claim is
# meaningless unless the graph it is computed on is pinned down.
THRESHOLD = 0.35     # |rho| above which an edge exists
WINDOW = 1250        # trading days of correlation history (~5 years)
SHOCK = 0.30         # initial distress applied to the shocked node
ABSOLUTE = True      # strong negative correlation still transmits stress

#: Sector labels, used only for the figure and for reporting — not by the model.
SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "INTC": "Technology",
    "IBM": "Technology", "ORCL": "Technology", "TXN": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "AXP": "Financials",
    "JNJ": "Health care", "PFE": "Health care", "MRK": "Health care",
    "UNH": "Health care",
    "XOM": "Energy", "CVX": "Energy", "SLB": "Energy",
    "PG": "Staples", "KO": "Staples", "WMT": "Staples", "MCD": "Discretionary",
    "CAT": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "NEE": "Utilities", "SO": "Utilities",
}


def centralities(graph: nx.Graph) -> pd.DataFrame:
    """Degree, eigenvector and betweenness centrality for every node.

    Eigenvector centrality is weighted by correlation strength; betweenness uses
    the Mantegna correlation *distance*, since a shortest path should run
    through strongly correlated pairs, and distance is what "short" means there.
    """
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))

    # Eigenvector centrality has no unique solution on a disconnected graph, and
    # these graphs are disconnected — thresholding leaves isolated nodes. It is
    # computed on the largest component and isolated nodes are scored zero,
    # which is the honest reading: a node reachable from nothing has no
    # eigenvector influence, and reporting NaN would just drop it from the rank
    # correlation the study is about.
    eigen = dict.fromkeys(graph.nodes(), 0.0)
    if graph.number_of_edges():
        largest = max(nx.connected_components(graph), key=len)
        component = graph.subgraph(largest)
        if component.number_of_nodes() > 2:
            with contextlib.suppress(nx.NetworkXException, ValueError):
                eigen.update(
                    nx.eigenvector_centrality_numpy(component, weight="weight")
                )

    between = nx.betweenness_centrality(graph, weight="distance")
    return pd.DataFrame(
        {
            "degree": pd.Series(degree),
            "weighted_degree": pd.Series(weighted_degree),
            "eigenvector": pd.Series(eigen),
            "betweenness": pd.Series(between),
        }
    )


def build(returns: pd.DataFrame, window: int, threshold: float) -> nx.Graph:
    return correlation_network(
        returns.iloc[-window:], method="threshold", threshold=threshold,
        absolute=ABSOLUTE,
    )


def analyse(
    returns: pd.DataFrame, window: int, threshold: float
) -> tuple[dict, pd.DataFrame | None, nx.Graph]:
    """One (window, threshold) pair: rank correlations and the disagreement."""
    graph = build(returns, window, threshold)
    if graph.number_of_edges() == 0:
        return {"window": window, "threshold": threshold, "edges": 0}, None, graph

    # rank_systemic_assets already reports degree; drop it so the join does not
    # collide, and take every centrality from one place instead.
    systemic = (
        rank_systemic_assets(graph, initial_distress=SHOCK)
        .set_index("asset")
        .drop(columns=["degree"])
    )
    table = systemic.join(centralities(graph))

    result = {
        "window": window,
        "threshold": threshold,
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
    }
    for measure in ("degree", "weighted_degree", "eigenvector", "betweenness"):
        rho, p = stats.spearmanr(table["DebtRank"], table[measure])
        result[f"spearman_{measure}"] = float(rho)
        result[f"spearman_{measure}_p"] = float(p)

    ranked_by_debtrank = table["DebtRank"].rank(ascending=False, method="min")
    ranked_by_degree = table["degree"].rank(ascending=False, method="min")
    most_connected = str(ranked_by_degree.idxmin())
    top_node = str(table["DebtRank"].idxmax())

    result["top_node"] = top_node
    result["top_node_degree_rank"] = int(ranked_by_degree[top_node])
    result["most_connected"] = most_connected
    result["most_connected_debtrank_rank"] = int(ranked_by_debtrank[most_connected])
    result["rank_gap"] = int(ranked_by_debtrank[most_connected]) - 1
    return result, table, graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    returns = to_returns(load_prices(UNIVERSE, start=args.start))
    tickers = list(returns.columns)

    print("=" * 78)
    print("1. UNIVERSE AND NETWORK CONSTRUCTION")
    print(f"   N_ASSETS : {len(tickers)}")
    print(f"   UNIVERSE : {', '.join(tickers)}")
    print(f"   Period   : {returns.index[0].date()} to {returns.index[-1].date()} "
          f"({len(returns)} days)")
    print(f"   WINDOW   : {WINDOW} trading days (the most recent slice)")
    print(f"   Edges    : |rho| > {THRESHOLD}, absolute={ABSOLUTE}, weight = |rho|")
    print(f"   DebtRank : initial distress {SHOCK:.0%}, equal node values "
          "(no market-cap weighting), each node propagates once then goes inactive")

    headline, table, graph = analyse(returns, WINDOW, THRESHOLD)

    print()
    print("=" * 78)
    print("2. SYSTEMIC RANK VS CONNECTIVITY")
    shown = table.sort_values("DebtRank", ascending=False)[
        ["DebtRank", "amplification", "degree", "weighted_degree",
         "eigenvector", "betweenness"]
    ].copy()
    shown.insert(0, "sector", [SECTORS.get(a, "?") for a in shown.index])
    with pd.option_context("display.width", 160, "display.max_rows", 40):
        print(shown.round(4).to_string())

    print()
    print("=" * 78)
    print("3. RANK_CORR — Spearman correlation with DebtRank")
    for measure in ("degree", "weighted_degree", "eigenvector", "betweenness"):
        print(f"   {measure:<18} rho = {headline[f'spearman_{measure}']:+.4f}   "
              f"p = {headline[f'spearman_{measure}_p']:.2e}")

    print()
    print("4-5. THE DISAGREEMENT")
    print(f"   TOP_NODE                    : {headline['top_node']}")
    print(f"   TOP_NODE_DEGREE_RANK        : {headline['top_node_degree_rank']}")
    print(f"   most connected              : {headline['most_connected']}")
    print(f"   DEGREE_TOP_DEBTRANK_RANK    : {headline['most_connected_debtrank_rank']}")

    print()
    print("6. CASCADE_NUMBER")
    top = debt_rank(graph, headline["top_node"], SHOCK)
    connected = debt_rank(graph, headline["most_connected"], SHOCK)
    print(f"   shock {headline['top_node']:<5} -> total loss {top.total_loss:.4%}  "
          f"DebtRank {top.debt_rank:.4f}  amplification {top.amplification:.2f}x")
    print(f"   shock {headline['most_connected']:<5} -> total loss "
          f"{connected.total_loss:.4%}  DebtRank {connected.debt_rank:.4f}  "
          f"amplification {connected.amplification:.2f}x")
    ratio = (top.total_loss / connected.total_loss) if connected.total_loss else float("nan")
    print(f"   ratio                       : {ratio:.4f}x")
    print(f"   difference                  : "
          f"{(top.total_loss - connected.total_loss):.4%} of portfolio value")

    print()
    print("=" * 78)
    print("SENSITIVITY — does the ordering survive different graphs?")
    rows = []
    for window in (500, 1250, len(returns)):
        for threshold in (0.25, 0.35, 0.45, 0.55):
            summary, _, _ = analyse(returns, min(window, len(returns)), threshold)
            rows.append(summary)
    grid = pd.DataFrame(rows)
    columns = ["window", "threshold", "edges", "density", "spearman_degree",
               "spearman_eigenvector", "spearman_betweenness", "top_node",
               "most_connected", "most_connected_debtrank_rank"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(grid[[c for c in columns if c in grid.columns]].round(4).to_string(index=False))

    print()
    print(f"   Spearman(degree, DebtRank) range: "
          f"{grid['spearman_degree'].min():+.4f} to {grid['spearman_degree'].max():+.4f}")
    flips = grid[grid["most_connected_debtrank_rank"] > 1]
    print(f"   Graphs where the most-connected asset is NOT the most systemic: "
          f"{len(flips)} of {len(grid)}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "contagion_centrality.csv")
    grid.to_csv(out / "contagion_sensitivity.csv", index=False)
    (out / "contagion_summary.json").write_text(json.dumps({
        "universe": tickers,
        "n_assets": len(tickers),
        "period": {"start": str(returns.index[0].date()),
                   "end": str(returns.index[-1].date())},
        "window": WINDOW,
        "threshold": THRESHOLD,
        "shock": SHOCK,
        "headline": headline,
        "cascade": {
            "top_node": headline["top_node"],
            "top_node_total_loss": top.total_loss,
            "most_connected": headline["most_connected"],
            "most_connected_total_loss": connected.total_loss,
            "ratio": ratio,
        },
        "sensitivity": grid.to_dict("records"),
    }, indent=2, default=str))
    print(f"\nWrote centrality table, sensitivity grid and summary JSON to {out}/")


if __name__ == "__main__":
    main()
