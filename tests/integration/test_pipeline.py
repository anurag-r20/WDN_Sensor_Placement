"""Integration tests: the notebook's pipeline end to end, with real samplers, on real network data.

    network files -> graph -> centrality -> Q matrix -> BQM -> sampler -> JSON log -> feasibility boundary

These tests are slower than the unit tests and are marked "integration"; run them alone with
`pytest -m integration`, or skip them with `pytest -m "not integration"`.

The last test checks the JSON logs committed under logs/ against the current code and the network data
in Data/WDN_Data. Set the WDN_BASE_DIR environment variable to read Data/ from another folder; the test is
skipped only if no Data/WDN_Data folder is found.
"""

from __future__ import annotations

import io
import itertools
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import neal
import numpy as np
import pytest
from enumeration_solver import placement_cost
from tabu import TabuSampler

import Utils

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parents[2]
# Folder that contains Data/WDN_Data for the logs regression test: WDN_BASE_DIR if set, else the repository root.
DATA_BASE_DIR = Path(os.environ.get("WDN_BASE_DIR", REPO_ROOT))
SENSORS: int = 5  # C(18, 5) = 8,568 placements: small enough to enumerate exactly
SEED: int = 7


@pytest.fixture
def test_network(test_network_base_dir):
    """The 18-node Test network as the notebook builds it (cells 3 to 7), with the notebook's rho."""
    with redirect_stdout(io.StringIO()):  # WDN_network_data prints every file path it reads
        coords, edges, water_consumption = Utils.WDN_network_data("Test", base_dir=test_network_base_dir)
    graph = Utils.Construct_Graph("Test", coords, edges)
    vertex_cost, edge_weight = Utils.centrality(graph, water_consumption)
    rho = float(np.sum(np.abs(list(vertex_cost.values()))) + 1)
    Q, cQ = Utils.build_Q_matrix(graph, vertex_cost, edge_weight, SENSORS, rho)
    return {
        "graph": graph,
        "vertex_cost": vertex_cost,
        "edge_weight": edge_weight,
        "rho": rho,
        "bqm": Utils.QUBO_dimod(Q, beta=cQ),
    }


@pytest.fixture
def exact_optimum(test_network) -> float:
    """The best objective with exactly SENSORS sensors, by enumerating every such placement."""
    nodes = list(test_network["graph"].nodes())
    best = float("inf")
    for chosen in itertools.combinations(nodes, SENSORS):
        sensors = {node: int(node in chosen) for node in nodes}
        best = min(best, placement_cost(test_network["vertex_cost"], test_network["edge_weight"], sensors))
    return best


@pytest.mark.parametrize(
    "run_sampler",
    [
        lambda bqm: neal.SimulatedAnnealingSampler().sample(bqm, num_reads=200, seed=SEED),
        lambda bqm: TabuSampler().sample(bqm, num_reads=10, timeout=20, seed=SEED),
    ],
    ids=["simulated_annealing", "tabu_search"],
)
def test__pipeline__given_the_test_network__sampler_returns_a_feasible_placement_no_better_than_the_optimum(
    test_network, exact_optimum, run_sampler
) -> None:
    # A heuristic may miss the optimum, but its best sample must use exactly s sensors (the penalty is
    # large enough) and can never beat the true optimum (if it did, the energy would be wrong).

    # Act
    best = run_sampler(test_network["bqm"]).first

    # Assert
    assert sum(best.sample.values()) == SENSORS
    assert best.energy >= exact_optimum - 1e-9


def test__pipeline__given_a_sampler_run_saved_to_json__reloads_it_and_finds_the_feasibility_boundary(
    test_network, tmp_path
) -> None:
    # Arrange
    sampleset = neal.SimulatedAnnealingSampler().sample(test_network["bqm"], num_reads=200, seed=SEED)
    qubo_parameters = {"rho": test_network["rho"], "s": SENSORS, "num_variables": 18}

    # Act
    Utils.save_results_json(
        {"results": {"min_energy": sampleset.first.energy}},
        "Test",
        "SimulatedAnnealing",
        str(tmp_path),
        sample_data=Utils.extract_sample_data(sampleset.aggregate()),
        qubo_parameters=qubo_parameters,
    )
    with redirect_stdout(io.StringIO()):
        result = Utils.load_and_plot_feasibility_from_json(
            str(tmp_path),
            "Test",
            solver_names=["SimulatedAnnealing"],
            save_path=str(tmp_path / "boundary.png"),
            show=False,
        )

    # Assert
    assert result["qubo_parameters"] == qubo_parameters
    assert result["feasibility_boundary"] == pytest.approx(sampleset.first.energy + test_network["rho"])
    assert (tmp_path / "boundary.png").exists()


def logged_runs() -> list:
    """Every JSON log under logs/ that records samples and QUBO parameters, as (city, path) pairs."""
    runs = []
    for path in sorted((REPO_ROOT / "logs").glob("*/*.json")):
        content = json.loads(path.read_text())
        if "samples" in content and "qubo_parameters" in content:
            runs.append(pytest.param(path.parent.name, path, id=f"{path.parent.name}/{path.name}"))
    return runs


@pytest.mark.skipif(not (DATA_BASE_DIR / "Data" / "WDN_Data").is_dir(), reason=f"no Data/WDN_Data in {DATA_BASE_DIR}")
@pytest.mark.parametrize("city,log_path", logged_runs())
def test__build_Q_matrix__given_a_logged_run__reproduces_the_logged_energies(city: str, log_path: Path) -> None:
    """Regression oracle: the published results must still follow from the current code and data.

    If centrality or build_Q_matrix changes, or the data files change, the energies in logs/ no longer
    describe the model the code builds, and this test says so.
    """
    # Arrange
    run = json.loads(log_path.read_text())
    with redirect_stdout(io.StringIO()):
        coords, edges, water_consumption = Utils.WDN_network_data(city, base_dir=str(DATA_BASE_DIR))
    graph = Utils.Construct_Graph(city, coords, edges)
    vertex_cost, edge_weight = Utils.centrality(graph, water_consumption)
    parameters = run["qubo_parameters"]

    # Act
    Q, cQ = Utils.build_Q_matrix(graph, vertex_cost, edge_weight, parameters["s"], parameters["rho"])

    # Assert
    assert parameters["num_variables"] == graph.number_of_nodes()
    for sample in run["samples"][:200]:
        x = np.array([sample["solution"][str(i)] for i in range(graph.number_of_nodes())], dtype=float)
        assert float(x @ Q @ x + cQ) == pytest.approx(sample["energy"], rel=1e-9, abs=1e-6)
