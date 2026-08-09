"""Chart and text-report generation.

Every figure here answers one question a risk report has to answer. Matplotlib
is used with the ``Agg`` backend so the module works headless, in CI, and inside
notebooks without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib


def _select_backend() -> None:
    """Force a headless backend only outside IPython.

    The module must work in CI and in plain scripts with no display, which
    requires "Agg". But forcing it inside a Jupyter kernel would override the
    inline backend and silently stop every figure from rendering, so IPython is
    left to manage its own backend.
    """
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return
    except ImportError:
        pass
    matplotlib.use("Agg")


_select_backend()

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .options import OptionType
from .var import RiskResult

__all__ = [
    "plot_greeks_profile",
    "plot_network",
    "plot_return_distribution",
    "plot_var_backtest",
    "results_table",
    "save_figure",
]

# A single palette so every figure in the report reads as one document.
PALETTE = {
    "primary": "#2b6cb0",
    "accent": "#dd6b20",
    "danger": "#c53030",
    "muted": "#718096",
    "grid": "#e2e8f0",
}


def _style(ax) -> None:
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_figure(fig, path: str | Path) -> Path:
    """Write a figure to ``path``, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def results_table(results: list[RiskResult]) -> pd.DataFrame:
    """Tabulate VaR/ES estimates side by side for comparison."""
    if not results:
        raise ValueError("no results to tabulate")
    return pd.DataFrame([r.as_row() for r in results])


def plot_return_distribution(returns, results: list[RiskResult], title: str | None = None):
    """Loss distribution with each method's VaR marked.

    Plotted on the *loss* axis (positive = money lost) so the tail under scrutiny
    is on the right, matching the sign convention used throughout the engine.
    """
    losses = -np.asarray(returns, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(losses, bins=80, color=PALETTE["primary"], alpha=0.55,
            edgecolor="white", linewidth=0.4, label="Realised daily losses")

    colors = [PALETTE["accent"], PALETTE["danger"], PALETTE["muted"], "#2f855a", "#6b46c1"]
    for i, res in enumerate(results):
        ax.axvline(
            res.var,
            color=colors[i % len(colors)],
            linestyle="--",
            linewidth=1.8,
            label=f"{res.method} VaR {res.confidence:.0%} = {res.var:.2%}",
        )

    ax.set_xlabel("Daily loss (positive = loss)")
    ax.set_ylabel("Frequency")
    ax.set_title(title or "Loss distribution and VaR estimates")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_var_backtest(result: BacktestResult):
    """Realised losses against the walk-forward VaR forecast, breaches flagged.

    This is the single most informative chart in the repository: it shows whether
    the model tracks changing volatility, and whether its failures cluster.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))
    losses = -result.realised_returns

    ax.plot(losses.index, losses.to_numpy(), color=PALETTE["muted"],
            linewidth=0.7, alpha=0.75, label="Realised loss")
    ax.plot(result.var_forecasts.index, result.var_forecasts.to_numpy(),
            color=PALETTE["primary"], linewidth=1.6,
            label=f"{result.confidence:.0%} VaR forecast ({result.method})")

    hits = result.breaches.astype(bool)
    if hits.any():
        ax.scatter(losses.index[hits], losses[hits], color=PALETTE["danger"],
                   s=26, zorder=5, label=f"Breaches ({int(hits.sum())})")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Walk-forward VaR backtest — {result.breach_rate:.2%} observed vs "
        f"{result.expected_rate:.2%} expected"
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_network(graph: nx.Graph, systemic: pd.DataFrame | None = None, seed: int = 42):
    """Correlation network, node size and colour driven by systemic importance."""
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(graph, seed=seed, weight="weight")

    # Both are deliberately polymorphic: matplotlib accepts either a scalar
    # applied to every node or a per-node sequence.
    sizes: float | np.ndarray
    colors: str | np.ndarray

    if systemic is not None and not systemic.empty:
        ranks = systemic.set_index("asset")["DebtRank"]
        values = np.array([float(ranks.get(n, 0.0)) for n in graph.nodes()])
        span = values.max() - values.min()
        normalised = (values - values.min()) / span if span > 0 else np.zeros_like(values)
        sizes = 400 + 1800 * normalised
        colors = values
    else:
        sizes, colors = 700.0, PALETTE["primary"]

    weights = [graph[u][v].get("weight", 0.5) for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=[0.4 + 3.0 * w for w in weights],
                           edge_color=PALETTE["grid"])
    nodes = nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes,
                                   node_color=colors, cmap="YlOrRd", edgecolors="#2d3748",
                                   linewidths=0.8)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9)

    if systemic is not None and not systemic.empty:
        fig.colorbar(nodes, ax=ax, label="DebtRank (systemic impact)", shrink=0.8)

    ax.set_title("Correlation network — node size and colour = systemic impact")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_greeks_profile(
    strike: float,
    expiry_years: float,
    rate: float,
    volatility: float,
    kind: OptionType = "call",
    spot_range: tuple[float, float] | None = None,
):
    """Price and Greeks across a range of spot prices.

    Delta, gamma and vega are plotted on a shared secondary axis; they live on
    comparable scales, whereas price does not, which is why the price gets its
    own axis rather than being squeezed alongside them.
    """
    from .options import black_scholes_greeks

    lo, hi = spot_range or (strike * 0.6, strike * 1.4)
    spots = np.linspace(lo, hi, 200)
    rows = [black_scholes_greeks(s, strike, expiry_years, rate, volatility, kind)
            for s in spots]

    fig, (ax_price, ax_greeks) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_price.plot(spots, [r.price for r in rows], color=PALETTE["primary"], linewidth=2)
    ax_price.axvline(strike, color=PALETTE["muted"], linestyle=":", linewidth=1.2)
    ax_price.set_ylabel("Option price")
    ax_price.set_title(
        f"{kind.capitalize()} — K={strike}, T={expiry_years}y, "
        f"r={rate:.1%}, sigma={volatility:.0%}"
    )
    _style(ax_price)

    ax_greeks.plot(spots, [r.delta for r in rows], label="Delta", color=PALETTE["accent"])
    ax_greeks.plot(spots, [r.gamma * 100 for r in rows], label="Gamma x100",
                   color=PALETTE["danger"])
    ax_greeks.plot(spots, [r.vega / 100 for r in rows], label="Vega (per vol pt)",
                   color="#2f855a")
    ax_greeks.axvline(strike, color=PALETTE["muted"], linestyle=":", linewidth=1.2)
    ax_greeks.set_xlabel("Spot price")
    ax_greeks.set_ylabel("Sensitivity")
    ax_greeks.legend(frameon=False, fontsize=9)
    _style(ax_greeks)

    fig.tight_layout()
    return fig
