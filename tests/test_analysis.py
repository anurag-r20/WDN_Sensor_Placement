"""Unit tests for src/analysis.py: turning samples into the tables and metrics the notebook plots
(probability tables, binning, time to solution, feasibility boundary, approximation gap).

Test names follow test__function__given_condition__expected_outcome. Tests marked xfail document a
confirmed bug listed in docs/code_review.md; delete the marker when the bug is fixed.
"""

from __future__ import annotations

import math

import dimod
import numpy as np
import pandas as pd
import pytest

from src import analysis


def probability_table(energy_to_probability: dict) -> pd.DataFrame:
    """Builds the table every function here consumes: Energy as the index, one Probability column."""
    table = pd.DataFrame({"Probability": list(energy_to_probability.values())}, index=list(energy_to_probability))
    table.index.name = "Energy"
    return table


# Three samples, two of which share an energy, in the format extract_sample_data writes to JSON.
SAMPLES = [
    {"solution": {"0": 1, "1": 0}, "energy": 2.0, "num_occurrences": 3},
    {"solution": {"0": 0, "1": 1}, "energy": 1.0, "num_occurrences": 1},
    {"solution": {"0": 1, "1": 1}, "energy": 2.0, "num_occurrences": 4},
]


# =============================================================================
# samples_to_df and sampleset_to_df
# =============================================================================


def test__samples_to_df__given_samples__returns_probabilities_that_sum_to_one() -> None:
    # Act
    table = analysis.samples_to_df(SAMPLES)

    # Assert
    assert table["Probability"].sum() == pytest.approx(1.0)


def test__samples_to_df__given_samples_sharing_an_energy__merges_them_into_one_row() -> None:
    # Act
    table = analysis.samples_to_df(SAMPLES)

    # Assert
    assert list(table.index) == [1.0, 2.0]  # sorted from lowest energy
    assert table.loc[2.0, "Probability"] == pytest.approx(7 / 8)
    assert table.index.name == "Energy"


def test__sampleset_to_df__given_the_same_samples_as_a_dict__agrees_with_samples_to_df() -> None:
    # Differential oracle: two functions in src/analysis.py build the same table from two input formats.

    # Arrange
    as_dict = {
        "energy": [sample["energy"] for sample in SAMPLES],
        "num_occurrences": [sample["num_occurrences"] for sample in SAMPLES],
    }

    # Act
    from_dict = analysis.sampleset_to_df(as_dict, imported=True)
    from_samples = analysis.samples_to_df(SAMPLES)

    # Assert
    pd.testing.assert_frame_equal(from_dict, from_samples)


def test__sampleset_to_df__given_a_dimod_sampleset__weights_each_energy_by_its_occurrences() -> None:
    # Arrange
    sampleset = dimod.SampleSet.from_samples(
        [{0: 1}, {0: 0}], vartype="BINARY", energy=[5.0, 0.0], num_occurrences=[1, 3]
    )

    # Act
    table = analysis.sampleset_to_df(sampleset)

    # Assert
    assert table.loc[0.0, "Probability"] == pytest.approx(0.75)
    assert table.loc[5.0, "Probability"] == pytest.approx(0.25)


# =============================================================================
# sum_infeas_soln
# =============================================================================


def test__sum_infeas_soln__given_energies_above_the_threshold__collapses_them_into_one_row_at_the_threshold() -> None:
    # Arrange
    table = probability_table({1.0: 0.5, 150.0: 0.3, 900.0: 0.2})

    # Act
    result = analysis.sum_infeas_soln(table, threshold=100)

    # Assert
    assert list(result.index) == [1.0, 100]
    assert result.loc[100, "Probability"] == pytest.approx(0.5)
    assert result["Probability"].sum() == pytest.approx(1.0)


@pytest.mark.xfail(reason="BUG-11: an energy equal to the threshold appears twice in the result index")
def test__sum_infeas_soln__given_an_energy_equal_to_the_threshold__returns_unique_energies() -> None:
    # Arrange
    table = probability_table({1.0: 0.5, 100.0: 0.5})

    # Act
    result = analysis.sum_infeas_soln(table, threshold=100)

    # Assert
    assert result.index.is_unique


# =============================================================================
# bin_energy_levels
# =============================================================================


def test__bin_energy_levels__given_energies_in_one_bin__merges_their_probabilities() -> None:
    # Arrange
    table = probability_table({1.01: 0.25, 1.04: 0.25, 2.55: 0.5})

    # Act
    binned = analysis.bin_energy_levels(table, bin_size=0.1)

    # Assert
    assert list(binned.index) == pytest.approx([1.0, 2.5])
    assert binned["Probability"].sum() == pytest.approx(1.0)


@pytest.mark.xfail(
    reason="BUG-08: floor division on floats puts edge values in the bin below (0.3 // 0.1 == 2.0)",
    raises=AssertionError,
)
@pytest.mark.parametrize("energy", [0.3, 0.7, 2.3])
def test__bin_energy_levels__given_an_energy_on_a_bin_edge__keeps_it_in_its_own_bin(energy: float) -> None:
    # Arrange
    table = probability_table({energy: 1.0})

    # Act
    binned = analysis.bin_energy_levels(table, bin_size=0.1)

    # Assert
    assert list(binned.index) == pytest.approx([energy])


# =============================================================================
# calculate_tts
# =============================================================================


def test__calculate_tts__given_half_the_reads_optimal__scales_the_run_time_to_99_percent_confidence() -> None:
    """Known oracle: TTS = t * log(1 - 0.99) / log(1 - p) = 2 * log(0.01) / log(0.5) for p = 0.5."""
    # Arrange
    samples = [({"a": 1, "b": 0}, 1.0, 5), ({"a": 0, "b": 1}, 3.0, 5)]

    # Act
    tts_opt, tts_fea, p_opt, p_fea = analysis.calculate_tts(samples, s=1, energy_threshold=1.0, exec_time=2.0)

    # Assert
    assert p_opt == pytest.approx(0.5)
    assert p_fea == pytest.approx(1.0)
    assert tts_opt == pytest.approx(2.0 * math.log(0.01) / math.log(0.5))
    assert tts_fea == pytest.approx(2.0)  # every read is feasible: one run is enough


def test__calculate_tts__given_no_read_reaches_the_threshold__returns_infinite_time() -> None:
    # Arrange
    samples = [({"a": 1, "b": 1}, 9.0, 10)]

    # Act
    tts_opt, tts_fea, p_opt, p_fea = analysis.calculate_tts(samples, s=1, energy_threshold=1.0, exec_time=2.0)

    # Assert
    assert p_opt == 0 and p_fea == 0
    assert math.isinf(tts_opt) and math.isinf(tts_fea)


def test__calculate_tts__given_a_dimod_sampleset__reads_samples_energies_and_occurrences() -> None:
    # Arrange
    sampleset = dimod.SampleSet.from_samples(
        [{"a": 1, "b": 0}, {"a": 1, "b": 1}], vartype="BINARY", energy=[1.0, 4.0], num_occurrences=[3, 1]
    )

    # Act
    _, _, p_opt, p_fea = analysis.calculate_tts(sampleset, s=1, energy_threshold=1.0, exec_time=1.0)

    # Assert
    assert p_opt == pytest.approx(0.75)
    assert p_fea == pytest.approx(0.75)


@pytest.mark.xfail(
    reason="BUG-06: for samples that are not dicts (e.g. numpy arrays) the sensor count is never "
    "recomputed, so every read counts as feasible",
    raises=AssertionError,
)
def test__calculate_tts__given_array_samples__counts_only_reads_with_s_sensors_as_feasible() -> None:
    # Arrange
    samples = [(np.array([1, 1, 0]), 1.0, 5), (np.array([1, 0, 0]), 2.0, 5)]

    # Act
    _, _, _, p_fea = analysis.calculate_tts(samples, s=2, energy_threshold=1.0, exec_time=1.0)

    # Assert
    assert p_fea == pytest.approx(0.5)


# =============================================================================
# calculate_feasibility_boundary
# =============================================================================


def test__calculate_feasibility_boundary__given_tables__returns_the_lowest_energy_plus_rho() -> None:
    # Arrange
    tables = [probability_table({3.0: 0.5, 9.0: 0.5}), probability_table({2.0: 1.0})]

    # Act
    boundary = analysis.calculate_feasibility_boundary(rho=4.0, s=2, df_list=tables)

    # Assert
    assert boundary == pytest.approx(6.0)


def test__calculate_feasibility_boundary__given_no_tables__returns_rho_alone() -> None:
    # Act
    boundary = analysis.calculate_feasibility_boundary(rho=4.0, s=2)

    # Assert
    assert boundary == pytest.approx(4.0)


def test__calculate_feasibility_boundary__given_empirical_method__returns_the_middle_of_the_largest_gap() -> None:
    # Arrange: energies 1, 2, 10, 11: the largest gap is 2 -> 10, whose middle is 6.
    tables = [probability_table({1.0: 0.25, 2.0: 0.25, 10.0: 0.25, 11.0: 0.25})]

    # Act
    boundary = analysis.calculate_feasibility_boundary(rho=4.0, s=2, df_list=tables, method="empirical")

    # Assert
    assert boundary == pytest.approx(6.0)


def test__calculate_feasibility_boundary__given_both_methods__returns_both_values() -> None:
    # Arrange
    tables = [probability_table({1.0: 0.25, 2.0: 0.25, 10.0: 0.25, 11.0: 0.25})]

    # Act
    result = analysis.calculate_feasibility_boundary(rho=4.0, s=2, df_list=tables, method="both")

    # Assert
    assert result["theoretical"] == pytest.approx(5.0)
    assert result["empirical"] == pytest.approx(6.0)
    assert result["boundaries_per_solver"] == pytest.approx([6.0])


@pytest.mark.parametrize("method,tables", [("guess", None), ("empirical", None), ("empirical", [])])
def test__calculate_feasibility_boundary__given_an_unusable_request__raises_value_error(method, tables) -> None:
    # Act
    with pytest.raises(Exception) as error:
        analysis.calculate_feasibility_boundary(rho=4.0, s=2, df_list=tables, method=method)

    # Assert
    assert isinstance(error.value, ValueError)


# =============================================================================
# calculate_approximation_ratio
# =============================================================================


@pytest.mark.parametrize(
    "glob_obj,best_obj,expected",
    [
        (100.0, 100.0, 0.0),  # matching the reference gives a gap of zero
        (150.0, 100.0, -0.5),  # the docstring claims 1.5 for this input
        (50.0, 100.0, 0.5),
    ],
)
def test__calculate_approximation_ratio__given_two_objectives__returns_one_minus_their_ratio(
    glob_obj: float, best_obj: float, expected: float
) -> None:
    # Characterization test: pins what the notebook plots as "1 - Approximation Ratio" (cell 14).

    # Act
    result = analysis.calculate_approximation_ratio(glob_obj, best_obj)

    # Assert
    assert result == pytest.approx(expected)


def test__calculate_approximation_ratio__given_a_zero_reference__raises_value_error() -> None:
    # Act
    with pytest.raises(Exception) as error:
        analysis.calculate_approximation_ratio(1.0, 0.0)

    # Assert
    assert isinstance(error.value, ValueError)
