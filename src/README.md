# `src`

The functions used by `WDN_Sensor_Placement_Latest.ipynb`, the tests and the scripts. They were split out
of the former `Utils.py`, one module per step of the pipeline:

```text
network files -> network -> formulations -> (Gurobi / samplers) -> results_io -> analysis -> plotting
                                         \-> experiments (sweeps over s and rho)
```

The notebook should only orchestrate and display: helper functions belong here, not in notebook cells.

## Files

- `__init__.py`
  - Package marker. Import from the owning module, e.g. `from src.formulations import build_Q_matrix`.

- `network.py`
  - `WDN_network_data`: reads the nodes, pipes and water consumption of a network from
    `<base_dir>/Data/WDN_Data/<City>_WDN/`.
  - `Construct_Graph`: builds the `networkx` graph, with each node's coordinates in its `pos` attribute.
  - `centrality`: scores the graph into vertex costs `VC` (normalized demand plus a degree term) and edge
    weights `EB` (edge betweenness).

- `formulations.py`
  - `build_Q_matrix`: the QUBO matrix `Q` and constant `cQ` (penalty `rho * (sum x - s)^2`).
  - `QUBO_dimod`: wraps `Q` and `cQ` into a dimod `BinaryQuadraticModel` for the samplers.
  - `create_pyomo_model`, `MIQP`, `QUBO`: the Pyomo models solved by Gurobi.

- `experiments.py`
  - `coverage`: solves the MIQP for every sensor count `s` (the coverage curve).
  - `iterate_over_rho`: runs simulated annealing and tabu search over penalty values `rho`.
  - `sensor_placement_results`: reads the sensor placements back from solved Pyomo models.

- `analysis.py`
  - `samples_to_df`, `sampleset_to_df`: energy -> probability tables from saved samples or a dimod `SampleSet`.
  - `sum_infeas_soln`, `bin_energy_levels`: collapse and bin those tables for plotting.
  - `calculate_tts`, `calculate_feasibility_boundary`, `calculate_approximation_ratio`: the metrics.

- `results_io.py`
  - `create_log_context`: sends `print` output to a log file.
  - `extract_sample_data`, `save_results_json`, `load_samples_from_json`: write and read
    `<log_dir>/<city>_<solver>.json`, including the samples and the QUBO parameters of a run.

- `plotting.py`
  - `plot_WDN`, `plot_sensor_placement`, `plot_comparison`: the network and its sensors.
  - `plot_energies`, `plot_samples`, `plot_enumerate`, `plot_multibar_graph_discrete`: sampler output.
  - `plot_coverage`, `plot_tts_comparison`: the coverage curve and the TTS comparison.
  - `load_and_plot_feasibility_from_json`: loads saved runs and plots the feasibility boundary.
  - `save_current_figure`: saves the current matplotlib figure.

## Dependencies between modules

`network`, `formulations`, `analysis` and `results_io` import nothing from `src`. `experiments` uses
`formulations` and `analysis`; `plotting` uses `analysis` and `results_io`. Keep it that way: a module
should never import from one further down the pipeline.

## Known issues

Confirmed bugs are listed in `docs/code_review.md` and pinned by tests marked `xfail` in `tests/`. The split
moved the code without changing behaviour, so every bug listed there is still present.
