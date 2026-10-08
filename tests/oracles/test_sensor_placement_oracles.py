"""Tests for the sensor placement model (build_Q_matrix, QUBO_dimod, QUBO, MIQP), sorted by test oracle.

A test oracle is whatever tells the test the expected behavior. Four kinds appear here:
- Known oracle: the expected answer, worked out by hand before the test runs.
- Differential oracle: several implementations of the same energy must agree on every placement.
- Pseudo-oracle: EnumerationSolver, a brute-force solver that tries every placement.
- Metamorphic relation: two related runs whose results must compare in a way you can prove, even when
  neither result is known.

The promises under test come from the notebook's formulation (cells 9 and 22):
  energy(x) = sum_i c_i x_i + sum_(i,j) w_ij (1 - x_i)(1 - x_j) + rho (sum_i x_i - s)^2
            = x^T Q x + cQ
and, for a large enough rho, the lowest-energy placement uses exactly s sensors and is the optimum of
the constrained problem.

Test names follow test__function__given_condition__expected_outcome, so a failing test reports in plain
words which promise was broken. Energies are compared with pytest.approx, never with ==.
"""

from __future__ import annotations

import itertools
import random

import dimod
import networkx as nx
import numpy as np
import pyomo.environ as pyo
import pytest
from enumeration_solver import EnumerationSolver, enumerate_binary_pyomo_model, placement_cost

from src import formulations, network

# A fixed seed makes the random sweep generate the same networks on every run, so a failure can be
# reproduced and debugged.
SWEEP_SEED: int = 20261007
SWEEP_INSTANCES: int = 100


def qubo_energy(Q: np.ndarray, cQ: float, bits) -> float:
    """Returns x^T Q x + cQ, the energy the samplers minimize, for one placement."""
    x = np.asarray(bits, dtype=float)
    return float(x @ Q @ x + cQ)


def notebook_rho(vertex_cost: dict) -> float:
    """The penalty the notebook uses (cells 17 and 23): the sum of |c_i| plus one."""
    return float(np.sum(np.abs(list(vertex_cost.values()))) + 1)


def random_network(rng: random.Random, min_nodes: int = 2, max_nodes: int = 8):
    """Returns (G, VC, EB, water_consumption) for a random connected network, scored with centrality."""
    n = rng.randint(min_nodes, max_nodes)
    while True:
        graph = nx.gnp_random_graph(n, 0.5, seed=rng.randrange(10**9))
        if nx.is_connected(graph):
            break
    graph = nx.relabel_nodes(graph, {i: f"J-{i}" for i in graph.nodes()})
    water_consumption = {node: rng.uniform(1.0, 100.0) for node in graph.nodes()}
    vertex_cost, edge_weight = network.centrality(graph, water_consumption)
    return graph, vertex_cost, edge_weight, water_consumption


@pytest.fixture
def kite_network():
    """A four-node network with a cycle and a dangling node, scored with centrality."""
    graph = nx.Graph()
    graph.add_edges_from([("J-1", "J-2"), ("J-2", "J-3"), ("J-3", "J-1"), ("J-3", "J-4")])
    water_consumption = {"J-1": 10.0, "J-2": 40.0, "J-3": 25.0, "J-4": 5.0}
    vertex_cost, edge_weight = network.centrality(graph, water_consumption)
    return graph, vertex_cost, edge_weight, water_consumption


# =============================================================================
# Known oracle
# =============================================================================


def test__build_Q_matrix__given_one_pipe_between_two_nodes__matches_the_hand_derived_matrix() -> None:
    """Known oracle: Q and cQ worked out by hand from the formulation in notebook cell 22.

    c = (1, 2), w_ab = 0.5, s = 1, rho = 3:
      Q_aa = c_a - w + rho - 2 rho s = 1 - 0.5 + 3 - 6 = -2.5
      Q_bb = c_b - w + rho - 2 rho s = 2 - 0.5 + 3 - 6 = -1.5
      Q_ab = Q_ba = w / 2 + rho = 0.25 + 3 = 3.25
      cQ   = rho s^2 + w = 3 + 0.5 = 3.5
    """
    # Arrange
    graph = nx.Graph([("a", "b")])
    vertex_cost = {"a": 1.0, "b": 2.0}
    edge_weight = {("a", "b"): 0.5}

    # Act
    Q, cQ = formulations.build_Q_matrix(graph, vertex_cost, edge_weight, s=1, rho=3.0)

    # Assert
    assert Q == pytest.approx(np.array([[-2.5, 3.25], [3.25, -1.5]]))
    assert cQ == pytest.approx(3.5)


def test__enumeration_solver__given_the_three_node_pipe__places_the_one_sensor_in_the_middle() -> None:
    """Checks the pseudo-oracle itself against a known answer before trusting it in the sweep below.

    With equal node costs, a sensor in the middle b covers both pipes (cost 1); a sensor at either end
    leaves the far pipe uncovered (cost 1 + 1 = 2).
    """
    # Arrange
    vertex_cost = {"a": 1.0, "b": 1.0, "c": 1.0}
    edge_weight = {("a", "b"): 1.0, ("b", "c"): 1.0}

    # Act
    placement = EnumerationSolver().run(vertex_cost, edge_weight, s=1)

    # Assert
    assert placement.sensors == {"a": 0, "b": 1, "c": 0}
    assert placement.objective_value == pytest.approx(1.0)


# =============================================================================
# Differential oracle: four implementations of the same energy
# =============================================================================


@pytest.mark.parametrize("s,rho", [(0, 1.0), (2, 2.5), (4, 10.0)])
def test__qubo_energy__given_every_placement_of_a_small_network__agrees_across_four_implementations(
    kite_network, s: int, rho: float
) -> None:
    """The Q matrix, the dimod BQM, the Pyomo QUBO objective and the formulation must give one energy.

    The notebook compares Gurobi (solving the Pyomo model) with samplers (solving the BQM built from Q).
    That comparison only means something if all of them encode the same function.
    """
    # Arrange
    graph, vertex_cost, edge_weight, water_consumption = kite_network
    nodes = list(graph.nodes())
    Q, cQ = formulations.build_Q_matrix(graph, vertex_cost, edge_weight, s, rho)
    bqm = formulations.QUBO_dimod(Q, beta=cQ)
    pyomo_model = formulations.QUBO(
        graph, formulations.create_pyomo_model(graph, water_consumption, vertex_cost, edge_weight, "kite"), s, rho
    )

    for bits in itertools.product((0, 1), repeat=len(nodes)):
        sensors = dict(zip(nodes, bits))
        for node, bit in sensors.items():
            pyomo_model.x[node].value = bit

        # Act
        from_definition = placement_cost(vertex_cost, edge_weight, sensors) + rho * (sum(bits) - s) ** 2
        from_q_matrix = qubo_energy(Q, cQ, bits)
        from_bqm = bqm.energy(dict(enumerate(bits)))
        from_pyomo = pyo.value(pyomo_model.obj)

        # Assert
        assert from_q_matrix == pytest.approx(from_definition), f"Q matrix, placement {bits}"
        assert from_bqm == pytest.approx(from_definition), f"BQM, placement {bits}"
        assert from_pyomo == pytest.approx(from_definition), f"Pyomo QUBO, placement {bits}"


# =============================================================================
# Pseudo-oracle: brute force on many random networks
# =============================================================================


def test__qubo_ground_state__given_random_small_networks_and_the_notebook_rho__is_the_constrained_optimum() -> (
    None
):
    """Pseudo-oracle: on random networks, the QUBO's lowest energy is the constrained optimum.

    This is the promise that justifies the notebook's choice rho = sum|c_i| + 1: with it, no infeasible
    placement can undercut the best feasible one. dimod's ExactSolver finds the true ground state of the
    BQM; EnumerationSolver finds the constrained optimum straight from the problem definition.
    """
    rng = random.Random(SWEEP_SEED)

    for _ in range(SWEEP_INSTANCES):
        # Arrange
        graph, vertex_cost, edge_weight, _ = random_network(rng)
        s = rng.randint(0, graph.number_of_nodes())
        rho = notebook_rho(vertex_cost)
        Q, cQ = formulations.build_Q_matrix(graph, vertex_cost, edge_weight, s, rho)
        bqm = formulations.QUBO_dimod(Q, beta=cQ)

        # Act
        ground_state = dimod.ExactSolver().sample(bqm).first
        reference = EnumerationSolver().run(vertex_cost, edge_weight, s)

        # Assert
        assert sum(ground_state.sample.values()) == s, f"infeasible ground state, s={s}, edges={edge_weight}"
        assert ground_state.energy == pytest.approx(reference.objective_value), f"s={s}, edges={edge_weight}"


# =============================================================================
# Metamorphic relations
# =============================================================================


def test__build_Q_matrix__given_the_nodes_in_reverse_order__has_the_same_ground_state_energy(kite_network) -> None:
    """Metamorphic relation: the order in which nodes are listed must not change the optimum."""
    # Arrange
    graph, vertex_cost, edge_weight, _ = kite_network
    reversed_graph = nx.Graph()
    reversed_graph.add_nodes_from(reversed(list(graph.nodes())))
    reversed_graph.add_edges_from(graph.edges())
    s, rho = 2, notebook_rho(vertex_cost)

    # Act
    Q, cQ = formulations.build_Q_matrix(graph, vertex_cost, edge_weight, s, rho)
    Q_rev, cQ_rev = formulations.build_Q_matrix(reversed_graph, vertex_cost, edge_weight, s, rho)
    original = dimod.ExactSolver().sample(formulations.QUBO_dimod(Q, cQ)).first.energy
    reordered = dimod.ExactSolver().sample(formulations.QUBO_dimod(Q_rev, cQ_rev)).first.energy

    # Assert
    assert reordered == pytest.approx(original)


@pytest.mark.parametrize("k", [0.5, 3.0, 100.0])
def test__build_Q_matrix__given_costs_weights_and_rho_multiplied_by_k__multiplies_every_energy_by_k(
    kite_network, k: float
) -> None:
    """Metamorphic relation: rescaling the whole problem rescales every energy and keeps the optimum."""
    # Arrange
    graph, vertex_cost, edge_weight, _ = kite_network
    s, rho = 2, 4.0
    scaled_cost = {node: k * value for node, value in vertex_cost.items()}
    scaled_weight = {edge: k * value for edge, value in edge_weight.items()}

    # Act
    Q, cQ = formulations.build_Q_matrix(graph, vertex_cost, edge_weight, s, rho)
    Q_k, cQ_k = formulations.build_Q_matrix(graph, scaled_cost, scaled_weight, s, k * rho)

    # Assert
    for bits in itertools.product((0, 1), repeat=graph.number_of_nodes()):
        assert qubo_energy(Q_k, cQ_k, bits) == pytest.approx(k * qubo_energy(Q, cQ, bits)), f"placement {bits}"


def test__build_Q_matrix__given_a_feasible_placement__has_an_energy_independent_of_rho(kite_network) -> None:
    """Metamorphic relation: the penalty is zero when exactly s sensors are placed, whatever rho is."""
    # Arrange
    graph, vertex_cost, edge_weight, _ = kite_network
    s = 2
    feasible = [bits for bits in itertools.product((0, 1), repeat=graph.number_of_nodes()) if sum(bits) == s]

    # Act
    Q_low, cQ_low = formulations.build_Q_matrix(graph, vertex_cost, edge_weight, s, rho=1.0)
    Q_high, cQ_high = formulations.build_Q_matrix(graph, vertex_cost, edge_weight, s, rho=50.0)

    # Assert
    for bits in feasible:
        assert qubo_energy(Q_high, cQ_high, bits) == pytest.approx(qubo_energy(Q_low, cQ_low, bits)), f"{bits}"


def test__build_Q_matrix__given_an_infeasible_placement__has_an_energy_that_grows_with_rho(kite_network) -> None:
    """Metamorphic relation: raising rho makes every placement with the wrong sensor count dearer."""
    # Arrange
    graph, vertex_cost, edge_weight, _ = kite_network
    s = 2
    infeasible = [bits for bits in itertools.product((0, 1), repeat=graph.number_of_nodes()) if sum(bits) != s]

    # Act
    Q_low, cQ_low = formulations.build_Q_matrix(graph, vertex_cost, edge_weight, s, rho=1.0)
    Q_high, cQ_high = formulations.build_Q_matrix(graph, vertex_cost, edge_weight, s, rho=50.0)

    # Assert
    for bits in infeasible:
        assert qubo_energy(Q_high, cQ_high, bits) > qubo_energy(Q_low, cQ_low, bits), f"placement {bits}"


def test__MIQP__given_a_larger_sensor_budget__does_not_raise_the_optimum(kite_network) -> None:
    """Metamorphic relation: the coverage curve (notebook cell 12) can only fall as s grows.

    MIQP() constrains sum x <= s, so a larger s only adds feasible placements. The notebook picks the
    "optimal number of sensors" as the first s where this curve reaches its minimum.
    """
    # Arrange
    graph, vertex_cost, edge_weight, water_consumption = kite_network
    base = formulations.create_pyomo_model(graph, water_consumption, vertex_cost, edge_weight, "kite")

    # Act
    optima = [enumerate_binary_pyomo_model(formulations.MIQP(graph, base, s)).objective_value for s in range(5)]

    # Assert
    for smaller_budget, larger_budget in zip(optima, optima[1:]):
        assert larger_budget <= smaller_budget + 1e-12


# =============================================================================
# Known bugs (see docs/code_review.md). Each test states the promise; xfail is removed with the fix.
# =============================================================================


@pytest.mark.xfail(
    reason="BUG-03: MIQP() constrains sum x <= s, but the documented model (cell 9) and the QUBO use = s",
    raises=AssertionError,
)
def test__MIQP__given_a_sensor_budget__has_the_same_optimum_as_the_documented_equality_model() -> None:
    """Known oracle: with costly sensors, '= 2' must place two sensors (cost 10.0); '<= 2' places none."""
    # Arrange
    graph = nx.Graph([("a", "b"), ("b", "c")])
    vertex_cost = {"a": 5.0, "b": 5.0, "c": 5.0}
    edge_weight = {("a", "b"): 0.1, ("b", "c"): 0.1}
    water_consumption = {"a": 1.0, "b": 1.0, "c": 1.0}
    base = formulations.create_pyomo_model(graph, water_consumption, vertex_cost, edge_weight, "pipe")

    # Act
    miqp_optimum = enumerate_binary_pyomo_model(formulations.MIQP(graph, base, 2))
    documented_optimum = EnumerationSolver().run(vertex_cost, edge_weight, s=2, relation="=")

    # Assert
    assert documented_optimum.objective_value == pytest.approx(10.0)
    assert miqp_optimum.objective_value == pytest.approx(documented_optimum.objective_value)
