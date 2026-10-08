"""Unit tests for src/formulations.py: the QUBO matrix, its dimod BQM and the Pyomo models.

Whether these models encode the sensor placement problem is tested in
tests/oracles/test_sensor_placement_oracles.py. This file tests the smaller promises of each function:
shapes, inputs it rejects, and side effects.

Test names follow test__function__given_condition__expected_outcome. Tests marked xfail document a
confirmed bug listed in docs/code_review.md; delete the marker when the bug is fixed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pyomo.environ as pyo
import pytest

from src import formulations


# =============================================================================
# build_Q_matrix
# =============================================================================


def test__build_Q_matrix__given_a_network__returns_a_symmetric_matrix_with_one_row_per_node(path_graph_abc) -> None:
    # Arrange
    vertex_cost = {"a": 1.0, "b": 2.0, "c": 3.0}
    edge_weight = {("a", "b"): 0.4, ("b", "c"): 0.6}

    # Act
    Q, _ = formulations.build_Q_matrix(path_graph_abc, vertex_cost, edge_weight, s=1, rho=5.0)

    # Assert
    assert Q.shape == (3, 3)
    assert Q == pytest.approx(Q.T)


@pytest.mark.xfail(reason="BUG-12: a self-loop's linear term is subtracted twice (-2w instead of -w)")
def test__build_Q_matrix__given_a_self_loop__encodes_its_edge_term_once() -> None:
    """Known oracle: for binary x, w (1 - x)(1 - x) = w (1 - x), so the energy at x = 1 must be 0."""
    # Arrange
    graph = nx.Graph()
    graph.add_node("a")
    vertex_cost = {"a": 0.0}
    edge_weight = {("a", "a"): 1.0}

    # Act
    Q, cQ = formulations.build_Q_matrix(graph, vertex_cost, edge_weight, s=0, rho=0.0)

    # Assert
    energy_with_sensor = Q[0, 0] + cQ
    assert energy_with_sensor == pytest.approx(0.0)


# =============================================================================
# QUBO_dimod
# =============================================================================


def test__QUBO_dimod__given_a_symmetric_matrix__merges_both_off_diagonal_entries_into_one_coupling() -> None:
    """dimod adds Q[0,1] and Q[1,0] into a single coupling, which is why build_Q_matrix stores w/2 twice.

    Note: the docstring of QUBO_dimod shows a different (wrong) result for this same input.
    """
    # Arrange
    Q = np.array([[1.0, -1.0], [-1.0, 2.0]])

    # Act
    bqm = formulations.QUBO_dimod(Q, beta=0.5)

    # Assert
    assert dict(bqm.linear) == pytest.approx({0: 1.0, 1: 2.0})
    assert bqm.quadratic[(0, 1)] == pytest.approx(-2.0)
    assert bqm.offset == pytest.approx(0.5)


# =============================================================================
# create_pyomo_model, MIQP, QUBO
# =============================================================================


def test__create_pyomo_model__given_a_network__has_one_binary_variable_per_node(path_graph_abc) -> None:
    # Arrange
    demand = {"a": 1.0, "b": 2.0, "c": 3.0}
    vertex_cost = {"a": 0.1, "b": 0.2, "c": 0.3}
    edge_weight = {("a", "b"): 0.5, ("b", "c"): 0.5}

    # Act
    model = formulations.create_pyomo_model(path_graph_abc, demand, vertex_cost, edge_weight, "pipe")

    # Assert
    assert sorted(model.x.keys()) == ["a", "b", "c"]
    assert all(model.x[node].domain is pyo.Binary for node in model.nodes)
    assert pyo.value(model.c["b"]) == pytest.approx(0.2)
    assert pyo.value(model.w[("a", "b")]) == pytest.approx(0.5)


def test__create_pyomo_model__given_no_graph__returns_none() -> None:
    # Act
    model = formulations.create_pyomo_model(None, {}, {}, {}, "Nowhere")

    # Assert
    assert model is None


@pytest.mark.parametrize("build", [lambda g, m: formulations.MIQP(g, m, 1), lambda g, m: formulations.QUBO(g, m, 1, 2.0)])
def test__MIQP_and_QUBO__given_a_base_model__leave_the_base_model_unchanged(path_graph_abc, build) -> None:
    """Both functions clone the base model, so the notebook can reuse it for every s (cell 12)."""
    # Arrange
    edge_weight = {("a", "b"): 0.5, ("b", "c"): 0.5}
    base = formulations.create_pyomo_model(path_graph_abc, {n: 1.0 for n in "abc"}, {n: 1.0 for n in "abc"}, edge_weight, "x")

    # Act
    built = build(path_graph_abc, base)

    # Assert
    assert hasattr(built, "obj")
    assert not hasattr(base, "obj")


@pytest.mark.parametrize("build", [lambda m: formulations.MIQP(None, m, 1), lambda m: formulations.QUBO(None, m, 1, 2.0)])
def test__MIQP_and_QUBO__given_no_base_model__return_none(build) -> None:
    # Act
    result = build(None)

    # Assert
    assert result is None
