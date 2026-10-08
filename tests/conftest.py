"""Shared fixtures for the test suite.

Every test module imports Utils, so this file makes sure that import is safe on a machine with no
display: matplotlib must use its non-interactive "Agg" backend BEFORE pyplot is imported anywhere.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import networkx as nx  # noqa: E402
import pytest  # noqa: E402

# tests/resources mirrors the layout WDN_network_data expects under its base_dir:
# <base_dir>/Data/WDN_Data/Test_WDN/Test_*.txt  (a copy of the 18-node "Test" network).
RESOURCES_DIR = Path(__file__).parent / "resources"


@pytest.fixture(autouse=True)
def close_all_figures():
    """Closes every matplotlib figure after each test, so figures never leak between tests."""
    yield
    plt.close("all")


@pytest.fixture
def test_network_base_dir() -> str:
    """The base_dir to pass to WDN_network_data to read the 18-node Test network under tests/resources."""
    return str(RESOURCES_DIR)


@pytest.fixture
def path_graph_abc() -> nx.Graph:
    """The three-node pipe a - b - c, with coordinates so the plotting functions can draw it."""
    graph = nx.Graph()
    graph.add_node("a", pos=(0.0, 0.0))
    graph.add_node("b", pos=(1.0, 0.0))
    graph.add_node("c", pos=(2.0, 0.0))
    graph.add_edges_from([("a", "b"), ("b", "c")])
    return graph
