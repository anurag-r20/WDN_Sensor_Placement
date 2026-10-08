"""Unit tests for the LOGGING UTILITIES section of Utils.py: writing solver runs to disk and reading
them back. Every test writes under pytest's tmp_path, never under the repository's logs/ folder.

Test names follow test__function__given_condition__expected_outcome. Tests marked xfail document a
confirmed bug listed in docs/code_review.md; delete the marker when the bug is fixed.
"""

from __future__ import annotations

import json
import sys

import dimod
import matplotlib.pyplot as plt
import numpy as np
import pytest

import Utils


def write_solver_json(folder, city: str, solver: str, energies: list, rho: float = 4.0, s: int = 1) -> None:
    """Writes one solver run in the format save_results_json produces, one sample per energy."""
    samples = [{"solution": {"0": 1}, "energy": energy, "num_occurrences": 1} for energy in energies]
    Utils.save_results_json(
        {"results": {}},
        city,
        solver,
        str(folder),
        sample_data=samples,
        qubo_parameters={"rho": rho, "s": s, "num_variables": 1},
    )


# =============================================================================
# create_log_context
# =============================================================================


def test__create_log_context__given_prints_inside_the_block__writes_them_to_the_log_file(tmp_path) -> None:
    # Arrange
    log_path = tmp_path / "run.log"

    # Act
    with Utils.create_log_context(str(log_path)):
        print("energy = 1.5")

    # Assert
    assert log_path.read_text() == "energy = 1.5\n"


def test__create_log_context__given_an_error_inside_the_block__still_restores_stdout(tmp_path) -> None:
    # Arrange
    stdout_before = sys.stdout

    # Act
    with pytest.raises(RuntimeError):
        with Utils.create_log_context(str(tmp_path / "run.log")):
            raise RuntimeError("sampler crashed")

    # Assert
    assert sys.stdout is stdout_before


# =============================================================================
# extract_sample_data, save_results_json, load_samples_from_json
# =============================================================================


def test__extract_sample_data__given_a_sampleset__returns_json_ready_samples() -> None:
    # Arrange
    sampleset = dimod.SampleSet.from_samples(
        [{0: 1, 1: 0}, {0: 0, 1: 1}], vartype="BINARY", energy=[1.5, 2.5], num_occurrences=[3, 1]
    )

    # Act
    samples = Utils.extract_sample_data(sampleset)

    # Assert
    assert samples == [
        {"solution": {"0": 1, "1": 0}, "energy": 1.5, "num_occurrences": 3},
        {"solution": {"0": 0, "1": 1}, "energy": 2.5, "num_occurrences": 1},
    ]
    json.dumps(samples)  # raises if anything is not JSON-serializable


def test__save_results_json__given_numpy_values__writes_them_as_plain_json_numbers(tmp_path) -> None:
    # Arrange
    results = {"results": {"min_energy": np.float64(1.5), "num_samples": np.int64(4000), "x": np.array([1, 0])}}

    # Act
    path = Utils.save_results_json(results, "Fossolo", "TabuSearch", str(tmp_path))

    # Assert
    assert path.endswith("Fossolo_TabuSearch.json")
    saved = json.loads((tmp_path / "Fossolo_TabuSearch.json").read_text())
    assert saved["results"] == {"min_energy": 1.5, "num_samples": 4000, "x": [1, 0]}
    assert saved["city"] == "Fossolo" and saved["solver"] == "TabuSearch"


@pytest.mark.xfail(reason="BUG-10: numpy booleans are not converted, so json.dump raises TypeError", raises=TypeError)
def test__save_results_json__given_a_numpy_boolean__writes_it_as_a_json_boolean(tmp_path) -> None:
    # Act
    Utils.save_results_json({"results": {"converged": np.bool_(True)}}, "Fossolo", "TabuSearch", str(tmp_path))

    # Assert
    saved = json.loads((tmp_path / "Fossolo_TabuSearch.json").read_text())
    assert saved["results"]["converged"] is True


def test__load_samples_from_json__given_a_saved_run__returns_the_samples_and_qubo_parameters(tmp_path) -> None:
    # Arrange
    write_solver_json(tmp_path, "Apulia", "SimulatedAnnealing", energies=[1.0, 2.0], rho=4.5, s=11)

    # Act
    data = Utils.load_samples_from_json(str(tmp_path / "Apulia_SimulatedAnnealing.json"))

    # Assert
    assert [sample["energy"] for sample in data["samples"]] == [1.0, 2.0]
    assert data["sample_metadata"] == {"total_unique_samples": 2, "total_occurrences": 2}
    assert data["qubo_parameters"] == {"rho": 4.5, "s": 11, "num_variables": 1}


@pytest.mark.parametrize("missing_key", ["samples", "qubo_parameters"])
def test__load_samples_from_json__given_a_file_without_a_required_key__raises_value_error(
    tmp_path, missing_key: str
) -> None:
    # Older runs in logs/Apulia were saved before samples and qubo_parameters were recorded.

    # Arrange
    content = {"samples": [], "qubo_parameters": {}}
    del content[missing_key]
    path = tmp_path / "old_run.json"
    path.write_text(json.dumps(content))

    # Act
    with pytest.raises(Exception) as error:
        Utils.load_samples_from_json(str(path))

    # Assert
    assert isinstance(error.value, ValueError)


# =============================================================================
# load_and_plot_feasibility_from_json
# =============================================================================


def test__load_and_plot_feasibility_from_json__given_two_solver_runs__returns_lowest_energy_plus_rho(tmp_path) -> None:
    # Arrange
    write_solver_json(tmp_path, "Apulia", "SimulatedAnnealing", energies=[2.0, 3.0, 9.0], rho=4.0)
    write_solver_json(tmp_path, "Apulia", "TabuSearch", energies=[1.5, 2.0], rho=4.0)
    plot_path = tmp_path / "boundary.png"

    # Act
    result = Utils.load_and_plot_feasibility_from_json(
        str(tmp_path), "Apulia", solver_names=["SimulatedAnnealing", "TabuSearch"], save_path=str(plot_path), show=False
    )

    # Assert
    assert result["feasibility_boundary"] == pytest.approx(5.5)
    assert result["solver_names"] == ["Simulated Annealing", "Tabu Search"]
    assert plot_path.exists()


def test__load_and_plot_feasibility_from_json__given_no_matching_files__raises_value_error(tmp_path) -> None:
    # Act
    with pytest.raises(Exception) as error:
        Utils.load_and_plot_feasibility_from_json(str(tmp_path), "Apulia", show=False)

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
    Utils.save_current_figure(str(path))

    # Assert
    assert path.exists() and path.stat().st_size > 0


def test__save_current_figure__given_an_empty_figure__writes_nothing(tmp_path) -> None:
    # This is what happens after a plotting function has already called plt.show() (see BUG-04).

    # Arrange
    path = tmp_path / "plot.png"

    # Act
    Utils.save_current_figure(str(path))

    # Assert
    assert not path.exists()
