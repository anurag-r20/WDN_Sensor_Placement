"""Tests for migrate_logs.py, which moved flat logs/<city>_<solver>.* files into logs/<city>/ folders.

The second test is a contract test between two modules: files the notebook saves, once migrated, must
still be readable by the function that plots them.

Test names follow test__function__given_condition__expected_outcome. Tests marked xfail document a
confirmed bug listed in docs/code_review.md; delete the marker when the bug is fixed.
"""

from __future__ import annotations

import migrate_logs
import pytest

from src import plotting, results_io


def test__migrate_logs_to_city_folders__given_flat_city_files__moves_each_into_its_city_folder(tmp_path) -> None:
    # Arrange
    for name in ["Fossolo_TabuSearch.json", "Fossolo_TabuSearch.log", "ZJ_LeapHybrid.json"]:
        (tmp_path / name).write_text("{}")

    # Act
    migrate_logs.migrate_logs_to_city_folders(str(tmp_path))

    # Assert
    assert sorted(p.name for p in (tmp_path / "Fossolo").iterdir()) == ["TabuSearch.json", "TabuSearch.log"]
    assert [p.name for p in (tmp_path / "ZJ").iterdir()] == ["LeapHybrid.json"]
    assert not any(p.is_file() for p in tmp_path.iterdir())


@pytest.mark.xfail(
    reason="BUG-02: migrate_logs strips the '<city>_' prefix but save_results_json and "
    "load_and_plot_feasibility_from_json expect it, so migrated runs cannot be loaded "
    "(this is the state of logs/Fossolo today)",
    raises=ValueError,
)
def test__migrate_logs_to_city_folders__given_saved_solver_runs__keeps_them_loadable_for_plotting(tmp_path) -> None:
    # Arrange
    for solver, energies in [("SimulatedAnnealing", [2.0, 3.0]), ("TabuSearch", [1.5])]:
        samples = [{"solution": {"0": 1}, "energy": e, "num_occurrences": 1} for e in energies]
        results_io.save_results_json(
            {}, "Fossolo", solver, str(tmp_path), sample_data=samples, qubo_parameters={"rho": 4.0, "s": 1}
        )

    # Act
    migrate_logs.migrate_logs_to_city_folders(str(tmp_path))
    result = plotting.load_and_plot_feasibility_from_json(
        str(tmp_path / "Fossolo"),
        "Fossolo",
        solver_names=["SimulatedAnnealing", "TabuSearch"],
        save_path=str(tmp_path / "boundary.png"),
        show=False,
    )

    # Assert
    assert result["feasibility_boundary"] == pytest.approx(5.5)
