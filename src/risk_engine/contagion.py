"""Correlation-network construction and DebtRank contagion.

Why this replaces an ad-hoc "loss spreads to neighbours" simulation
-------------------------------------------------------------------
Two failure modes make a naive contagion script meaningless:

1. **A complete graph with random edge weights.** If every asset is joined to
   every other with a random number, the network encodes no information and the
   simulation output is a function of the random seed alone. Here the graph is
   built from the *empirical* correlation matrix and then sparsified — by
   threshold or by minimum spanning tree — so an edge means something.

2. **Unbounded repeated shocks.** Multiplying a node's value by ``(1 - loss)``
   every round, forever, guarantees eventual collapse and measures nothing.
   DebtRank (Battiston et al., 2012) fixes this with a state machine: a node
   propagates its distress exactly once, then stops. That single rule is what
   makes the resulting number a bounded, comparable measure of systemic impact
   instead of an artefact of the round count.

The output, ``DebtRank``, is the fraction of total portfolio value destroyed by a
shock, excluding the initially shocked nodes' own direct loss — that is, the
*amplification* the network contributes.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "ContagionResult",
    "correlation_network",
    "debt_rank",
    "rank_systemic_assets",
]


@dataclass(frozen=True)
class ContagionResult:
    """Outcome of one DebtRank propagation."""

    shocked: list[str]
    debt_rank: float
    direct_loss: float
    total_loss: float
    final_distress: dict[str, float]
    rounds: int

    @property
    def amplification(self) -> float:
        """Total loss divided by direct loss: how much the network multiplies a shock."""
        return self.total_loss / self.direct_loss if self.direct_loss > 0 else 0.0

    def summary(self) -> str:
        return (
            f"shock={','.join(self.shocked)}  direct={self.direct_loss:.4f}  "
            f"total={self.total_loss:.4f}  DebtRank={self.debt_rank:.4f}  "
            f"amplification={self.amplification:.2f}x  rounds={self.rounds}"
        )


def correlation_network(
    returns: pd.DataFrame,
    method: str = "threshold",
    threshold: float = 0.3,
    absolute: bool = True,
) -> nx.Graph:
    """Build a weighted graph from the empirical correlation matrix.

    Parameters
    ----------
    method
        ``"threshold"`` keeps edges whose correlation exceeds ``threshold``.
        ``"mst"`` keeps the minimum spanning tree over the correlation distance
        ``d = sqrt(2(1 - rho))``, the standard Mantegna construction — it yields a
        connected backbone with exactly ``n-1`` edges and no threshold to tune.
        ``"complete"`` keeps every edge, retained only so the sparsification
        effect can be measured against it.
    absolute
        Treat strong negative correlation as a strong link. Appropriate for
        contagion, where an inverse relationship still transmits stress.
    """
    if returns.shape[1] < 2:
        raise ValueError("need at least two assets to build a network")

    corr = returns.corr()
    if corr.isna().to_numpy().any():
        raise ValueError("correlation matrix contains NaN; check for constant series")
    names = list(corr.columns)

    graph = nx.Graph()
    graph.add_nodes_from(names)

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rho = float(corr.loc[a, b])
            weight = abs(rho) if absolute else max(rho, 0.0)
            if method == "threshold" and weight < threshold:
                continue
            if weight <= 0.0:
                continue
            # `distance` is the Mantegna metric, used only by the MST branch.
            graph.add_edge(a, b, weight=weight, correlation=rho,
                           distance=float(np.sqrt(2.0 * (1.0 - rho))))

    if method == "mst":
        full = nx.Graph()
        full.add_nodes_from(names)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                rho = float(corr.loc[a, b])
                full.add_edge(a, b, weight=abs(rho) if absolute else max(rho, 0.0),
                              correlation=rho,
                              distance=float(np.sqrt(2.0 * (1.0 - rho))))
        graph = nx.minimum_spanning_tree(full, weight="distance")
    elif method not in ("threshold", "complete"):
        raise ValueError(f"unknown network method {method!r}")

    return graph


def _impact_matrix(graph: nx.Graph, names: list[str], values: np.ndarray) -> np.ndarray:
    """Value-weighted impact of node ``j`` on node ``i``.

    Correlation gives the *strength* of transmission; relative portfolio value
    gives its *scale*. A highly correlated but tiny position cannot destabilise a
    large one, so the edge weight is scaled by the source's share of the target's
    value and capped at 1.
    """
    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((n, n))
    for a, b, data in graph.edges(data=True):
        i, j = index[a], index[b]
        w = float(data.get("weight", 0.0))
        matrix[i, j] = min(w * values[j] / values[i], 1.0) if values[i] > 0 else 0.0
        matrix[j, i] = min(w * values[i] / values[j], 1.0) if values[j] > 0 else 0.0
    return matrix


def debt_rank(
    graph: nx.Graph,
    shocked: list[str] | str,
    initial_distress: float = 0.3,
    values: dict[str, float] | None = None,
    max_rounds: int = 100,
) -> ContagionResult:
    """Run DebtRank propagation from an initial shock.

    Each node holds a distress level in ``[0, 1]`` and one of three states:
    *undistressed*, *distressed* (propagating), *inactive* (already propagated).
    A node transmits its distress to neighbours exactly once, then goes inactive.
    That termination rule is the whole point — it bounds the cascade, so the
    result measures network structure rather than iteration count.
    """
    if isinstance(shocked, str):
        shocked = [shocked]
    names = list(graph.nodes())
    missing = set(shocked) - set(names)
    if missing:
        raise ValueError(f"shocked assets not in network: {sorted(missing)}")
    if not 0.0 < initial_distress <= 1.0:
        raise ValueError(f"initial_distress must lie in (0, 1], got {initial_distress}")

    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    value_vec = np.array(
        [1.0 if values is None else float(values.get(name, 1.0)) for name in names]
    )
    if (value_vec <= 0).any():
        raise ValueError("asset values must be strictly positive")
    economic_weight = value_vec / value_vec.sum()

    impact = _impact_matrix(graph, names, value_vec)

    distress = np.zeros(n)
    # 0 = undistressed, 1 = distressed (will propagate), 2 = inactive (done).
    state = np.zeros(n, dtype=int)
    for name in shocked:
        distress[index[name]] = initial_distress
        state[index[name]] = 1

    direct_loss = float(np.sum(distress * economic_weight))

    rounds = 0
    for rounds in range(1, max_rounds + 1):
        active = np.flatnonzero(state == 1)
        if active.size == 0:
            rounds -= 1
            break

        # Distress accumulates from every currently-propagating neighbour, and is
        # capped at 1 — a node cannot lose more than all of its value.
        incoming = impact[:, active] @ distress[active]
        new_distress = np.minimum(distress + incoming, 1.0)

        newly_hit = (new_distress > distress + 1e-12) & (state == 0)
        distress = new_distress
        state[active] = 2
        state[newly_hit] = 1

    total_loss = float(np.sum(distress * economic_weight))
    return ContagionResult(
        shocked=list(shocked),
        debt_rank=total_loss - direct_loss,
        direct_loss=direct_loss,
        total_loss=total_loss,
        final_distress={name: float(distress[index[name]]) for name in names},
        rounds=rounds,
    )


def rank_systemic_assets(
    graph: nx.Graph,
    initial_distress: float = 0.3,
    values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Shock each asset in isolation and rank by the damage it causes.

    This is the actionable output: the top of this table is where concentration
    limits and hedges belong. Degree centrality is reported alongside to show
    that systemic importance is *not* simply the most-connected node — value and
    correlation strength both matter.
    """
    degrees = dict(graph.degree())
    rows = []
    for name in graph.nodes():
        result = debt_rank(graph, name, initial_distress, values)
        rows.append(
            {
                "asset": name,
                "DebtRank": result.debt_rank,
                "total_loss": result.total_loss,
                "amplification": result.amplification,
                "degree": degrees[name],
                "rounds": result.rounds,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("DebtRank", ascending=False)
        .reset_index(drop=True)
    )
