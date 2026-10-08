# Code review: WDN sensor placement (`WDN_Sensor_Placement`)

*Reviewed 2026-10-07 against `Utils.py`, `WDN_Sensor_Placement_Latest.ipynb`, `INP_TO_TXT.ipynb`,
`migrate_logs.py` and the JSON logs under `logs/`. Every bug below was reproduced by running the code.
Most have a matching test marked `xfail` in `tests/`.*

## 1. Objectives

The project chooses where to put pressure sensors in a water distribution network (WDN) and compares
classical and quantum solvers on that problem.

- **Model.** The network is a graph. Each node *i* has a cost *c_i* (normalized demand plus a degree term).
  Each pipe *(i, j)* has a weight *w_ij* (edge betweenness). The objective
  `min Σ c_i x_i + Σ w_ij (1 − x_i)(1 − x_j)` with exactly `s` sensors (Speziali et al., 2021) rewards
  covering important pipes with few, cheap sensors.
- **Formulations.** An MIQP solved by Gurobi through Pyomo, and a QUBO that moves the sensor-count
  constraint into a penalty `ρ(Σx − s)²`.
- **Solvers compared.** Gurobi (MIQP and QUBO), simulated annealing (`neal`), tabu search, D-Wave quantum
  annealing (Advantage 6.4) and the Leap hybrid solver.
- **Metrics.** Energy distributions, the feasibility boundary (which samples satisfy `Σx = s`), the
  approximation gap to Gurobi, and time-to-solution (TTS) at 99% confidence, as a function of network size
  (24 to 856 nodes).

## 2. Current state of deliverables

| Network | Nodes | SA | Tabu | QA (feasible reads) | Leap | Feasibility plot | Notes |
|---|---:|:-:|:-:|:-:|:-:|:-:|---|
| Apulia | 24 | ✓ | ✓ | ✓ (3.8%) | ✓ | ✓ | 4 stale duplicate JSONs in the old format, without samples |
| Fossolo | 37 | ✓ | ✓ | ✓ (32.5%) | ✓ | ✓* | *Plot predates the migration. The files can no longer be loaded (BUG-02) |
| ZJ | 114 | ✓ | ✓ | ✓ (**0.0%**) | ✓ | ✓ | QA best energy is 10,444, against 4.87 for Tabu |
| Modena | 272 | ✓ | ✓ | ✗ | ✗ | ✗ | QA embedding was interrupted. `s = 87` is hard-coded (cell 23) |
| Kentucky | ~856 | ✗ | ✗ | ✗ | ✗ | ✗ | Data and loader entry exist, but nothing has been run |
| Pescara | 74 | – | – | – | – | – | Data in `Data/WDN_Data` but not wired into `WDN_network_data` |

**Gurobi baselines.** Gurobi results (MIQP or QUBO) are not logged for any network. The approximation
gaps and the TTS table in cell 52 are typed by hand, so they cannot be traced back to a run.

**Solution quality from the logs.**

| Network | Best feasible energy (SA / Tabu / Leap / QA) |
|---|---|
| Apulia | 1.502 / **1.300** / 1.475 / 1.969 |
| Fossolo | 2.084 / **1.859** / 2.076 / 2.494 |
| ZJ | 6.851 / **4.871** / 6.463 / no feasible sample |
| Modena | 15.51 / **8.008** / not run / not run |

Tabu search wins on every network.

**Integrity check.** Every energy logged in the 14 complete JSON files is reproduced, to within 1e-6, by
today's `centrality` and `build_Q_matrix` on the data in `Data/WDN_Data`. The published numbers
therefore still match the code. This is now an integration test (section 6).

**Reproducibility.** The refactor that added `Utils.py` and `logs/` (0a702ac) was reverted on `main`
(f1a7c0e); this branch re-applies it. The `benchmark` kernel the notebook was run with no longer exists.
`Data/` is committed, but cell 3 reads it through a hard-coded absolute path to a local folder instead of the
repository root. The notebook was run out of order: 14 code cells, including all the Gurobi and TTS cells,
did not run in the last session. The output of cell 46 comes from ZJ while cell 3 selects Modena.

## 3. Bug report

Severity: **H** gives wrong results or crashes a documented step. **M** gives wrong output in some cases or
breaks reproducibility. **L** is an edge case.

| ID | Sev | Where | What happens | Test |
|---|:-:|---|---|---|
| BUG-01 | H | `Utils.py:1189` `iterate_over_rho` | Reads `G`, `VC`, `EB` and `objective_value_list_MIP_min` as globals that exist only in the notebook, so it raises `NameError`. Cell 14 (the ρ sweep) cannot run. | `test_utils_optimization.py` |
| BUG-02 | H | `migrate_logs.py:41` vs `Utils.py:89,331` | Saving and loading both use `logs/<city>/<city>_<solver>.json`, while the migration strips the `<city>_` prefix. As a result, `load_and_plot_feasibility_from_json('logs/Fossolo', 'Fossolo')` raises *"No valid JSON files found"* today. Apulia keeps both versions. | `test_migrate_logs.py` |
| BUG-03 | H | `Utils.py:1409` `MIQP` | The constraint is `Σx ≤ s`, but the documented model (cell 9) and the QUBO penalty use `Σx = s`. The two coincide only at the `s` where the coverage curve first reaches its minimum. Elsewhere Gurobi and the samplers solve different problems: on a 3-node pipe with `s = 2` the optimum is 0.2 versus 10.0. | `oracles/test_sensor_placement_oracles.py` |
| BUG-04 | M | `Utils.py:1137` `plot_sensor_placement` | Calls `plt.show()` unconditionally. In Jupyter that closes the figure, so `plot_comparison(show=False)` followed by `save_current_figure` saves nothing (cell 21). | `test_utils_plotting.py` |
| BUG-05 | M | `Utils.py:1125` | Tests `== 1` on solver values. Gurobi can return `0.9999999`, and that sensor is then drawn as "no sensor". | `test_utils_plotting.py` |
| BUG-06 | M | `Utils.py:917-920` `calculate_tts` | When a sample is not dict-like (a NumPy array or list), `total_size` keeps its previous value, so every read counts as feasible: `p_fea = 1.0` instead of 0.5. | `test_utils_data_processing.py` |
| BUG-07 | M | Notebook cells 48–51 | TTS mixes units. SA and Tabu use wall-clock seconds, while Leap `run_time` and QA `qpu_access_time` are in µs. The cells also hard-code `s = 15/11` and thresholds 1.877/1.313 from other networks while `city = "Modena"` (s = 87). Check whether the published TTS values were converted. | none (notebook) |
| BUG-08 | L/M | `Utils.py:826` `bin_energy_levels` (also lines 707, 717) | Float floor division misplaces values on bin edges: `0.3 // 0.1 == 2.0`, so 0.3 lands in the 0.2 bin and 0.7 in the 0.6 bin. Bars, the Gurobi line and the boundary line can be off by one bin. | `test_utils_data_processing.py` |
| BUG-09 | L | `Utils.py:1531-1533` `centrality` | A node missing from the consumption file is silently dropped (the docstring promises `KeyError`). `build_Q_matrix` then fails later with an unrelated-looking `KeyError`. | `test_utils_network.py` |
| BUG-10 | L | `Utils.py:112-121` `save_results_json` | `np.bool_` is not converted, so `json.dump` raises `TypeError`. The function also mutates the caller's dict. | `test_utils_logging.py` |
| BUG-11 | L | `Utils.py:846-851` `sum_infeas_soln` | An energy exactly equal to the threshold produces a duplicate index entry, and the index name is dropped. | `test_utils_data_processing.py` |
| BUG-12 | L | `Utils.py:1068-1073` `build_Q_matrix` | For a self-loop `(i, i)` the linear term is subtracted twice (−2w instead of −w). | `test_utils_optimization.py` |
| BUG-13 | L | Notebook cell 24 | `G = nx.from_numpy_array(Q)` overwrites the network graph. Later uses of `G` for plotting or sensor results would be wrong. | none (notebook) |
| BUG-14 | L | Docstrings | Several docstrings contradict the code. `calculate_approximation_ratio` claims (150, 100) → 1.5 but returns −0.5. The `QUBO_dimod` example output is wrong. `build_Q_matrix` omits `rho`. `coverage` documents a `rho` argument it does not take. `WDN_network_data` says it raises, but it returns `(None, None, None)`. The `Construct_Graph` example uses the wrong signature. | characterization tests |

### Suspected issues (confirm intent)

- **SUSPECT-1, `centrality` degree term (`Utils.py:1534`).** `nx.degree_centrality` is already divided by
  *n − 1*, and the code divides by *n − 1* again. The degree term is therefore at most 1/(n−1), about 0.004
  for Modena, so the vertex cost is effectively demand alone. A characterization test pins the current
  behaviour.
- **SUSPECT-2, quantum annealing results.** QA feasibility falls from 32.5% (Fossolo) to 3.8% (Apulia) to
  0% (ZJ), and the Modena embedding never finished. The penalty `ρ(Σx − s)²` adds ρ to every off-diagonal
  entry, so the QUBO is a complete graph. That forces long qubit chains, and the large ρ swamps the
  objective. The logs do not record chain-break fraction or chain strength, so this cannot be diagnosed
  from the saved data.

## 4. Improvements (in priority order)

1. **Make the results reproducible.** Pin the environment (`requirements.txt` has been added). Call
   `WDN_network_data(city)` without `base_dir`, which defaults to the repository root where `Data/` is
   committed, instead of the absolute path in cell 3. Pass `seed=` to `neal` and `TabuSampler`, and record
   seeds and package versions in every JSON log. Use *Restart & Run All* before saving the notebook.
2. **Fix BUG-01 to BUG-03.** Pass the graph and the reference optimum into `iterate_over_rho`. Choose one
   log-naming scheme, then fix or delete `migrate_logs.py` and the stale Apulia duplicates. Decide whether
   the MIQP uses `=` or `≤` and make the code, the notebook text and the QUBO agree.
3. **Use a smaller, provable penalty.** For ρ > maxᵢ max(cᵢ, Σ_{j∈N(i)} w_ij − cᵢ), every infeasible
   placement can be moved one sensor at a time toward a feasible one with strictly lower energy, so the
   ground state is feasible. This held on 400 random networks, and for these networks it gives ρ ≈ 1.0,
   against 4.5 (Apulia) to 110 (Kentucky) for the notebook's `Σ|cᵢ| + 1`. A 4–110× smaller energy scale
   should help QA, given its limited coefficient precision, and SA. Re-run QA on ZJ with it.
4. **Split `Utils.py`.** It is 1,659 lines, and four functions are defined twice (`bin_energy_levels`,
   `sum_infeas_soln`, `sampleset_to_df`, `calculate_tts`; the second copy silently wins, so edits to the
   first have no effect). Split it into modules along its own section headers (`io`, `network`, `model`,
   `metrics`, `plotting`) and replace `from Utils import *` with explicit imports.
5. **Move the experiment out of the notebook.** A script `run_city.py --city ZJ --solvers sa,tabu,qa` would
   write one JSON per solver, including Gurobi baselines and QA chain-break statistics. The notebook would
   then only read logs and plot, and the TTS table would be computed from logs with consistent units.
6. **Keep plotting free of side effects.** Plot functions should return `fig, ax` and never call
   `plt.show()`, and `save_current_figure` should take the figure explicitly. Also reduce
   `figsize=(80, 60)` at 300 dpi (24,000 × 18,000 px) in `plot_comparison`.
7. **Smaller fixes.** Use the `logging` module instead of redirecting `stdout`. Stop mutating input dicts.
   Raise exceptions instead of printing and returning `None`. Read node and edge files deterministically,
   since `edges` is a `set`, so its order changes between runs. Fix the docstrings listed in BUG-14.
8. **Run the tests automatically.** Add a GitHub Actions workflow that runs `pytest -m "not integration"`
   on every push, as in SEFOP's `ci-unit-tests.yml`.

## 5. What was not checked

- Gurobi runs, because no license or solver was available. The MIQP and QUBO Pyomo models are tested by
  brute-force enumeration instead.
- QPU and Leap runs, which need a D-Wave token. Their logged results are verified against the model.
- `INP_TO_TXT.ipynb`. Its output is the data under `Data/WDN_Data`, which loads correctly.

## 6. Test suite (SEFOP conventions)

The tests follow [sefop/training-testing-python](https://github.com/sefop/training-testing-python):

- names read as `test__function__given_condition__expected_outcome`;
- each test has an Arrange / Act / Assert layout;
- floats are compared with `pytest.approx`, never `==`;
- the expected behaviour comes from an explicit oracle.

```
tests/
  conftest.py                         Agg backend, shared fixtures
  test_utils_network.py               WDN_network_data, Construct_Graph, centrality
  test_utils_optimization.py          build_Q_matrix, QUBO_dimod, Pyomo models, approximation ratio
  test_utils_data_processing.py       probability tables, binning, TTS, feasibility boundary
  test_utils_logging.py               log context, JSON save/load, figure saving
  test_utils_plotting.py              show=False contract and sensor colouring (mocks)
  test_migrate_logs.py                migration, and its contract with the JSON loader
  oracles/
    enumeration_solver.py             brute-force pseudo-oracle (shares no code with Utils)
    test_sensor_placement_oracles.py  known, differential, pseudo-oracle and metamorphic tests
  integration/test_pipeline.py        data -> QUBO -> SA/Tabu -> JSON -> boundary; logs regression
  resources/Data/WDN_Data/Test_WDN/   copy of the 18-node Test network
```

**Running the tests.**

```bash
pip install -r requirements.txt
pytest                                   # 85 passed, 13 xfailed
pytest -m "not integration"              # fast unit tests only
pytest -m integration                    # pipeline runs, and every JSON log in logs/ against Data/
WDN_BASE_DIR=/other/folder pytest        # read Data/ from another folder
```

**Known-bug tests.** Each confirmed bug has a test that states the correct behaviour and is marked
`@pytest.mark.xfail(reason="BUG-xx …")`. `pytest.ini` sets `xfail_strict = true`, so once a bug is fixed
its test reports *XPASS* and the run fails. Delete the marker at that point, and from then on the test
guards the fix.

**Checking the tests themselves.** Eight deliberate mutations were each applied to a copy of `Utils.py`,
and all eight were caught. They were:

- *w*/2 → *w* in Q;
- dropping Σw from cQ;
- 2ρ → ρ in the penalty;
- dropping the degree term;
- 99% → 90% in TTS;
- unnormalized probabilities;
- an unsquared Pyomo penalty;
- 2ρ in the boundary.
