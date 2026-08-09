"""Contagion tests.

The properties asserted here are exactly the ones an ad-hoc loss-propagation
loop fails: bounded output, guaranteed termination, and sensitivity to network
structure rather than to the number of iterations.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from risk_engine.contagion import correlation_network, debt_rank, rank_systemic_assets


@pytest.fixture
def block_returns() -> pd.DataFrame:
    """Two tightly-correlated blocks plus one independent asset.

    The structure is known, so the systemic ranking has a correct answer: the
    isolated asset must rank last.
    """
    rng = np.random.default_rng(4)
    n = 2000
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    def noise():
        return rng.standard_normal(n) * 0.35

    return pd.DataFrame(
        {
            "A1": f1 + noise(),
            "A2": f1 + noise(),
            "A3": f1 + noise(),
            "B1": f2 + noise(),
            "B2": f2 + noise(),
            "LONE": rng.standard_normal(n),
        }
    )


def test_threshold_network_is_sparser_than_complete(block_returns):
    complete = correlation_network(block_returns, method="complete")
    thresholded = correlation_network(block_returns, method="threshold", threshold=0.5)
    assert thresholded.number_of_edges() < complete.number_of_edges()
    # Every surviving edge must actually clear the threshold.
    assert all(d["weight"] >= 0.5 for _, _, d in thresholded.edges(data=True))


def test_mst_has_exactly_n_minus_one_edges_and_is_connected(block_returns):
    mst = correlation_network(block_returns, method="mst")
    assert mst.number_of_edges() == mst.number_of_nodes() - 1
    assert nx.is_connected(mst)


def test_threshold_network_recovers_the_block_structure(block_returns):
    """Within-block pairs must be linked; the independent asset must not be."""
    graph = correlation_network(block_returns, method="threshold", threshold=0.5)
    assert graph.has_edge("A1", "A2")
    assert graph.has_edge("B1", "B2")
    assert graph.degree("LONE") == 0


def test_debt_rank_is_bounded_by_total_portfolio_value(block_returns):
    """Distress is capped at 1 per node, so total loss can never exceed 100%."""
    graph = correlation_network(block_returns, method="complete")
    for shock in (0.1, 0.5, 1.0):
        result = debt_rank(graph, "A1", initial_distress=shock)
        assert 0.0 <= result.total_loss <= 1.0 + 1e-9
        assert result.debt_rank >= -1e-12


def test_debt_rank_terminates_well_before_the_round_cap(block_returns):
    """Each node propagates once, so the cascade length is bounded by the graph.

    This is the property the original multiplicative loop lacked — it would run
    for whatever round count it was given and never converge.
    """
    graph = correlation_network(block_returns, method="complete")
    result = debt_rank(graph, "A1", 0.4, max_rounds=100)
    assert result.rounds < 20


def test_isolated_node_causes_no_contagion(block_returns):
    """An unconnected asset can damage only itself, by definition."""
    graph = correlation_network(block_returns, method="threshold", threshold=0.5)
    result = debt_rank(graph, "LONE", 0.5)
    assert result.debt_rank == pytest.approx(0.0)
    assert result.amplification == pytest.approx(1.0)


def test_connected_node_amplifies_the_shock(block_returns):
    graph = correlation_network(block_returns, method="threshold", threshold=0.5)
    result = debt_rank(graph, "A1", 0.5)
    assert result.debt_rank > 0.0
    assert result.amplification > 1.0


def test_systemic_ranking_places_the_isolated_asset_last(block_returns):
    """The known-truth check on the ranking: LONE transmits nothing."""
    graph = correlation_network(block_returns, method="threshold", threshold=0.5)
    ranking = rank_systemic_assets(graph, initial_distress=0.4)
    assert ranking.iloc[-1]["asset"] == "LONE"
    assert ranking.iloc[0]["DebtRank"] > ranking.iloc[-1]["DebtRank"]


def test_larger_shock_causes_at_least_as_much_damage(block_returns):
    graph = correlation_network(block_returns, method="complete")
    small = debt_rank(graph, "A1", 0.2).total_loss
    large = debt_rank(graph, "A1", 0.6).total_loss
    assert large > small


def test_value_weighting_makes_large_positions_more_systemic(block_returns):
    """A big position transmits more distress than an identically-connected small one."""
    graph = correlation_network(block_returns, method="complete")
    equal = {n: 1.0 for n in graph.nodes()}
    concentrated = dict(equal, A1=10.0)
    assert (
        debt_rank(graph, "A1", 0.4, values=concentrated).total_loss
        > debt_rank(graph, "A1", 0.4, values=equal).total_loss
    )


def test_shocking_a_missing_asset_is_an_error(block_returns):
    graph = correlation_network(block_returns)
    with pytest.raises(ValueError, match="not in network"):
        debt_rank(graph, "NOPE", 0.3)


def test_single_asset_cannot_form_a_network():
    with pytest.raises(ValueError, match="at least two assets"):
        correlation_network(pd.DataFrame({"A": [0.01, 0.02, 0.03]}))


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_shock_rejected(block_returns, bad):
    graph = correlation_network(block_returns)
    with pytest.raises(ValueError, match="initial_distress"):
        debt_rank(graph, "A1", bad)
