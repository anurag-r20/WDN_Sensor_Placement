"""Unit tests for the PLOTTING FUNCTIONS section of Utils.py.

Pictures are hard to assert on, so these tests check the two promises the notebook relies on, using
mocks (stand-ins that record how they were called instead of drawing):
- with show=False a function must not call plt.show(), because in Jupyter plt.show() closes the
  figure and the save_current_figure call that follows it would then save nothing;
- a node is drawn as a sensor exactly when the solver placed one there.

Test names follow test__function__given_condition__expected_outcome. Tests marked xfail document a
confirmed bug listed in docs/code_review.md; delete the marker when the bug is fixed.
"""

from __future__ import annotations

import dimod
import pandas as pd
import pytest

import Utils


@pytest.fixture
def show_calls(monkeypatch) -> list:
    """Replaces plt.show with a mock and returns the list of calls it records."""
    calls: list = []
    monkeypatch.setattr(Utils.plt, "show", lambda *args, **kwargs: calls.append((args, kwargs)))
    return calls


@pytest.fixture
def drawn_node_colors(monkeypatch) -> list:
    """Replaces nx.draw_networkx_nodes with a mock and returns the node colors it was asked to draw."""
    colors: list = []
    monkeypatch.setattr(Utils.nx, "draw_networkx_nodes", lambda G, pos, **kwargs: colors.extend(kwargs["node_color"]))
    return colors


@pytest.fixture
def small_sampleset() -> dimod.SampleSet:
    return dimod.SampleSet.from_samples(
        [{0: 1, 1: 0}, {0: 0, 1: 1}], vartype="BINARY", energy=[1.0, 2.0], num_occurrences=[3, 1]
    )


PIPE_COSTS = {"a": 1.0, "b": 2.0, "c": 3.0}
PIPE_WEIGHTS = {("a", "b"): 0.5, ("b", "c"): 0.5}


def test__plot_WDN__given_show_false__does_not_call_plt_show(path_graph_abc, show_calls) -> None:
    # Act
    Utils.plot_WDN(path_graph_abc, PIPE_COSTS, PIPE_WEIGHTS, "Pipe", show=False)

    # Assert
    assert show_calls == []


def test__plot_energies__given_show_false__does_not_call_plt_show(small_sampleset, show_calls) -> None:
    # Act
    Utils.plot_energies(small_sampleset, title="SA", show=False)

    # Assert
    assert show_calls == []


def test__plot_enumerate__given_show_false__does_not_call_plt_show(small_sampleset, show_calls) -> None:
    # Act
    Utils.plot_enumerate(small_sampleset, title="SA", show=False)

    # Assert
    assert show_calls == []


def test__plot_multibar_graph_discrete__given_show_false__does_not_call_plt_show(show_calls) -> None:
    # Arrange
    table = pd.DataFrame({"Probability": [0.6, 0.4]}, index=pd.Index([1.0, 5.0], name="Energy"))

    # Act
    Utils.plot_multibar_graph_discrete([table], ["SA"], ["red"], feasibility_boundary=3.0, show=False)

    # Assert
    assert show_calls == []


@pytest.mark.xfail(
    reason="BUG-04: plot_sensor_placement always calls plt.show(), so plot_comparison(show=False) "
    "cannot leave its figure open for save_current_figure",
    raises=AssertionError,
)
def test__plot_comparison__given_show_false__does_not_call_plt_show(path_graph_abc, show_calls) -> None:
    # Act
    Utils.plot_comparison(path_graph_abc, PIPE_COSTS, PIPE_WEIGHTS, "Pipe", {"a": 1, "b": 0, "c": 0}, show=False)

    # Assert
    assert show_calls == []


def test__plot_sensor_placement__given_exact_zeros_and_ones__draws_sensors_in_red(
    path_graph_abc, show_calls, drawn_node_colors
) -> None:
    # Act
    Utils.plot_sensor_placement(path_graph_abc, PIPE_COSTS, PIPE_WEIGHTS, "Pipe", {"a": 1, "b": 0, "c": 1}, "t")

    # Assert
    assert drawn_node_colors == ["red", "skyblue", "red"]


@pytest.mark.xfail(
    reason="BUG-05: solver values such as 0.9999999 fail the '== 1' test, so the sensor is drawn as absent",
    raises=AssertionError,
)
def test__plot_sensor_placement__given_solver_values_within_tolerance_of_one__draws_sensors_in_red(
    path_graph_abc, show_calls, drawn_node_colors
) -> None:
    # Gurobi reports binaries as floats that can be off by about 1e-9 (see its IntFeasTol parameter).

    # Act
    Utils.plot_sensor_placement(
        path_graph_abc, PIPE_COSTS, PIPE_WEIGHTS, "Pipe", {"a": 0.9999999, "b": 1e-9, "c": 1.0000001}, "t"
    )

    # Assert
    assert drawn_node_colors == ["red", "skyblue", "red"]
