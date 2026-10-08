"""Unit tests for src/experiments.py: sweeps that run the solvers.

coverage and sensor_placement_results need Gurobi, which the test environment does not have; they are
covered indirectly by the Pyomo model tests in tests/oracles.

Test names follow test__function__given_condition__expected_outcome. Tests marked xfail document a
confirmed bug listed in docs/code_review.md; delete the marker when the bug is fixed.
"""

from __future__ import annotations

import pytest

from src import experiments


# =============================================================================
# iterate_over_rho
# =============================================================================


@pytest.mark.xfail(
    reason="BUG-01: iterate_over_rho reads G, VC, EB and objective_value_list_MIP_min as globals that only "
    "exist in the notebook, so it raises NameError",
    raises=NameError,
)
def test__iterate_over_rho__given_rho_values__returns_one_gap_per_rho_for_each_sampler() -> None:
    # Act
    simulated_annealing_gaps, tabu_gaps = experiments.iterate_over_rho(1, [1.0, 2.0])

    # Assert
    assert len(simulated_annealing_gaps) == 2
    assert len(tabu_gaps) == 2
