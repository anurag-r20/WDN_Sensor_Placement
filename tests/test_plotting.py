"""Unit tests for src/plotting.py.

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
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src import plotting


@pytest.fixture
def show_calls(monkeypatch) -> list:
    """Replaces plt.show with a mock and returns the list of calls it records."""
    calls: list = []
    monkeypatch.setattr(plotting.plt, "show", lambda *args, **kwargs: calls.append((args, kwargs)))
    return calls


@pytest.fixture
def drawn_node_colors(monkeypatch) -> list:
    """Replaces nx.draw_networkx_nodes with a mock and returns the node colors it was asked to draw."""
    colors: list = []
    monkeypatch.setattr(plotting.nx, "draw_networkx_nodes", lambda G, pos, **kwargs: colors.extend(kwargs["node_color"]))
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
    plotting.plot_WDN(path_graph_abc, PIPE_COSTS, PIPE_WEIGHTS, "Pipe", show=False)

    # Assert
    assert show_calls == []


def test__plot_energies__given_show_false__does_not_call_plt_show(small_sampleset, show_calls) -> None:
    # Act
    plotting.plot_energies(small_sampleset, title="SA", show=False)

    # Assert
    assert show_calls == []


def test__plot_enumerate__given_show_false__does_not_call_plt_show(small_sampleset, show_calls) -> None:
    # Act
    plotting.plot_enumerate(small_sampleset, title="SA", show=False)

    # Assert
    assert show_calls == []


def test__plot_multibar_graph_discrete__given_show_false__does_not_call_plt_show(show_calls) -> None:
    # Arrange
    table = pd.DataFrame({"Probability": [0.6, 0.4]}, index=pd.Index([1.0, 5.0], name="Energy"))

    # Act
    plotting.plot_multibar_graph_discrete([table], ["SA"], ["red"], feasibility_boundary=3.0, show=False)

    # Assert
    assert show_calls == []


@pytest.mark.xfail(
    reason="BUG-04: plot_sensor_placement always calls plt.show(), so plot_comparison(show=False) "
    "cannot leave its figure open for save_current_figure",
    raises=AssertionError,
)
def test__plot_comparison__given_show_false__does_not_call_plt_show(path_graph_abc, show_calls) -> None:
    # Act
    plotting.plot_comparison(path_graph_abc, PIPE_COSTS, PIPE_WEIGHTS, "Pipe", {"a": 1, "b": 0, "c": 0}, show=False)

    # Assert
    assert show_calls == []


def test__plot_sensor_placement__given_exact_zeros_and_ones__draws_sensors_in_red(
    path_graph_abc, show_calls, drawn_node_colors
) -> None:
    # Act
    plotting.plot_sensor_placement(path_graph_abc, PIPE_COSTS, PIPE_WEIGHTS, "Pipe", {"a": 1, "b": 0, "c": 1}, "t")

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
    plotting.plot_sensor_placement(
        path_graph_abc, PIPE_COSTS, PIPE_WEIGHTS, "Pipe", {"a": 0.9999999, "b": 1e-9, "c": 1.0000001}, "t"
    )

    # Assert
    assert drawn_node_colors == ["red", "skyblue", "red"]


# =============================================================================
# load_and_plot_feasibility_from_json
# =============================================================================


def test__load_and_plot_feasibility_from_json__given_two_solver_runs__returns_lowest_energy_plus_rho(tmp_path, write_solver_run) -> None:
    # Arrange
    write_solver_run(tmp_path, "Apulia", "SimulatedAnnealing", energies=[2.0, 3.0, 9.0], rho=4.0)
    write_solver_run(tmp_path, "Apulia", "TabuSearch", energies=[1.5, 2.0], rho=4.0)
    plot_path = tmp_path / "boundary.png"

    # Act
    result = plotting.load_and_plot_feasibility_from_json(
        str(tmp_path), "Apulia", solver_names=["SimulatedAnnealing", "TabuSearch"], save_path=str(plot_path), show=False
    )

    # Assert
    assert result["feasibility_boundary"] == pytest.approx(5.5)
    assert result["solver_names"] == ["Simulated Annealing", "Tabu Search"]
    assert plot_path.exists()


def test__load_and_plot_feasibility_from_json__given_no_matching_files__raises_value_error(tmp_path) -> None:
    # Act
    with pytest.raises(Exception) as error:
        plotting.load_and_plot_feasibility_from_json(str(tmp_path), "Apulia", show=False)

    # Assert
    assert isinstance(error.value, ValueError)


# =============================================================================
# save_current_figure
# =============================================================================


def test__save_current_figure__given_a_figure_with_content__writes_the_image(tmp_path) -> None:
    # Arrange
    plt.plot([0, 1], [1, 0])
    path = tmp_path / "plot.png"

    # Act
    plotting.save_current_figure(str(path))

    # Assert
    assert path.exists() and path.stat().st_size > 0


def test__save_current_figure__given_an_empty_figure__writes_nothing(tmp_path) -> None:
    # This is what happens after a plotting function has already called plt.show() (see BUG-04).

    # Arrange
    path = tmp_path / "plot.png"

    # Act
    plotting.save_current_figure(str(path))

    # Assert
    assert not path.exists()
