"""Unit tests for the NETWORK/GRAPH FUNCTIONS section of Utils.py: reading a network and scoring it.

The data under tests/resources is a copy of the 18-node "Test" network from Data/WDN_Data/Test_WDN.

Test names follow test__function__given_condition__expected_outcome. Tests marked xfail document a
confirmed bug listed in docs/code_review.md; delete the marker when the bug is fixed.
"""

from __future__ import annotations

import networkx as nx
import pytest

import Utils

# =============================================================================
# WDN_network_data
# =============================================================================


def test__WDN_network_data__given_the_test_network__reads_every_node_edge_and_consumption(
    test_network_base_dir,
) -> None:
    # Act
    coords, edges, water_consumption = Utils.WDN_network_data("Test", base_dir=test_network_base_dir)

    # Assert
    assert len(coords) == 18
    assert len(edges) == 18
    assert set(water_consumption) == set(coords)
    assert coords["J-1"] == pytest.approx((1597.24, 1180.0))
    assert water_consumption["J-2"] == pytest.approx(117.034)


def test__WDN_network_data__given_padded_names_in_the_files__strips_the_whitespace(test_network_base_dir) -> None:
    # The files are written as "J-3, J-15": without stripping, " J-15" would be a different node.

    # Act
    coords, edges, _ = Utils.WDN_network_data("Test", base_dir=test_network_base_dir)

    # Assert
    assert ("J-3", "J-15") in edges
    assert all(name == name.strip() for name in coords)


def test__WDN_network_data__given_an_unknown_city__returns_three_nones(test_network_base_dir) -> None:
    # Characterization test: the docstring promises a ValueError, but the code prints and returns Nones.

    # Act
    result = Utils.WDN_network_data("Atlantis", base_dir=test_network_base_dir)

    # Assert
    assert result == (None, None, None)


def test__WDN_network_data__given_a_base_dir_without_data__returns_three_nones(tmp_path) -> None:
    # Act
    result = Utils.WDN_network_data("Test", base_dir=str(tmp_path))

    # Assert
    assert result == (None, None, None)


# =============================================================================
# Construct_Graph
# =============================================================================


def test__Construct_Graph__given_coordinates_and_edges__stores_each_coordinate_as_the_node_position() -> None:
    # Arrange
    coords = {"a": (0.0, 1.0), "b": (2.0, 3.0)}
    edges = {("a", "b")}

    # Act
    graph = Utils.Construct_Graph("Tiny", coords, edges)

    # Assert
    assert nx.get_node_attributes(graph, "pos") == coords
    assert list(graph.edges()) == [("a", "b")]


@pytest.mark.parametrize("coords,edges", [(None, {("a", "b")}), ({"a": (0, 0)}, None)])
def test__Construct_Graph__given_missing_data__returns_none(coords, edges) -> None:
    # Act
    graph = Utils.Construct_Graph("Tiny", coords, edges)

    # Assert
    assert graph is None


# =============================================================================
# centrality
# =============================================================================


def test__centrality__given_a_three_node_pipe__weights_each_pipe_by_its_edge_betweenness(path_graph_abc) -> None:
    """Known oracle: of the 3 node pairs, 2 route through each pipe, so each edge betweenness is 2/3."""
    # Act
    _, edge_weight = Utils.centrality(path_graph_abc, {"a": 1.0, "b": 1.0, "c": 1.0})

    # Assert
    assert edge_weight == pytest.approx({("a", "b"): 2 / 3, ("b", "c"): 2 / 3})


def test__centrality__given_a_three_node_pipe__scores_nodes_by_normalized_demand_plus_scaled_degree(
    path_graph_abc,
) -> None:
    """Characterization test of the vertex cost c_i = f_i + g_i.

    f_i is the demand divided by the largest demand. g_i is networkx's degree centrality (already divided
    by n - 1) divided by n - 1 a second time; see SUSPECT-1 in docs/code_review.md. For the pipe a - b - c
    (n = 3), degree centrality is 0.5, 1.0, 0.5, so g = 0.25, 0.5, 0.25.
    """
    # Act
    vertex_cost, _ = Utils.centrality(path_graph_abc, {"a": 10.0, "b": 40.0, "c": 20.0})

    # Assert
    assert vertex_cost == pytest.approx({"a": 0.25 + 0.25, "b": 1.0 + 0.5, "c": 0.5 + 0.25})


def test__centrality__given_a_network__keys_the_edge_weights_like_the_graph_edges(path_graph_abc) -> None:
    # create_pyomo_model indexes its weight parameter by G.edges(), so the orientation must match.

    # Act
    _, edge_weight = Utils.centrality(path_graph_abc, {"a": 1.0, "b": 1.0, "c": 1.0})

    # Assert
    assert set(edge_weight) == set(path_graph_abc.edges())


@pytest.mark.xfail(
    reason="BUG-09: a node missing from the consumption data is silently dropped from the vertex costs "
    "(the docstring promises a KeyError); build_Q_matrix then fails far from the cause",
    raises=pytest.fail.Exception,
)
def test__centrality__given_a_node_without_consumption_data__raises_key_error(path_graph_abc) -> None:
    # Act
    with pytest.raises(Exception) as error:
        Utils.centrality(path_graph_abc, {"a": 1.0, "b": 2.0})

    # Assert
    assert isinstance(error.value, KeyError)
