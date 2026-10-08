"""Running the solvers over a sweep: sensor counts for the coverage curve, penalty values for the
QUBO, and reading sensor placements back from solved Pyomo models.
"""

import neal
import pyomo.environ as pyo
from tabu import TabuSampler

from .analysis import calculate_approximation_ratio
from .formulations import QUBO_dimod, build_Q_matrix


def coverage(G, s, model, MIQP, solver):
    """Computes and plots coverage metrics for the Water Distribution Network (WDN) using both MIP and QUBO formulations.
    This function iteratively solves the MIP and QUBO models for different numbers of sensors and records
    the objective values. It uses the Gurobi solver to find the optimal solutions and appends the results
    to lists for comparison.
    Args:
        G (nx.Graph): A NetworkX graph object representing the water distribution network (WDN) for the city.
        s (list): The list of sensor counts to try.
        rho (float): Penalty parameter for the QUBO model.
        model: The base Pyomo model to clone for each run.
        MIQP: The MIQP function to use for model construction.
        solver: The Pyomo solver instance to use.
    Returns:
        list: A list of objective values obtained from the MIP formulation.
    Raises:
        ValueError: If the solver or model is not properly defined.
    Examples:
        >>> objective_value_list_MIQP = coverage(G, s, rho, model, MIQP, solver)
        >>> print(objective_value_list_MIQP)
        [100, 95, 90]
    """
    objective_value_list_MIP = []
    for s_val in s:
        model_MIQP = MIQP(G, model, s_val)
        results_MIQP = solver.solve(model_MIQP, tee=True)
        objective_value_MIP = pyo.value(model_MIQP.obj)
        objective_value_list_MIP.append(objective_value_MIP)
    return objective_value_list_MIP


def sensor_placement_results(model_MIQP, model_QUBO):
    """Prints and returns sensor placement results for MIP and QUBO models.
    This function displays the sensor placement status for each node in both MIP and QUBO models. It provides
    a clear view of which nodes have sensors placed according to each optimization approach. It also returns
    dictionaries with sensor placement information for further analysis.
    Args:
        model_MIQP (pyo.ConcreteModel): The Pyomo model with MIP formulation, including sensor placement variables.
        model_QUBO (pyo.ConcreteModel): The Pyomo model with QUBO formulation, including sensor placement variables.
    Returns:
        tuple: A tuple containing two dictionaries:
            - sensor_placement_MIQP (dict): Dictionary with node names as keys and binary values (0 or 1) indicating
              whether a sensor is placed at each node according to the MIP model.
            - sensor_placement_QUBO (dict): Dictionary with node names as keys and binary values (0 or 1) indicating
              whether a sensor is placed at each node according to the QUBO model.
    Examples:
        >>> sensor_placement_MIQP, sensor_placement_QUBO = sensor_placement_results(model_MIQP, model_QUBO)
        >>> print(sensor_placement_MIQP)
        {'J1': 1, 'J2': 0, 'J3': 1}
        >>> print(sensor_placement_QUBO)
        {'J1': 1, 'J2': 1, 'J3': 0}
    """
    print("MIP results")
    for node in model_MIQP.nodes:
        print(f"Node {node}: Sensor placed = {pyo.value(model_MIQP.x[node])}")
    sensor_placement_MIQP = {node: pyo.value(model_MIQP.x[node]) for node in model_MIQP.nodes}
    print("\nQUBO results")
    for node in model_QUBO.nodes:
        print(f"Node {node}: Sensor placed = {pyo.value(model_QUBO.x[node])}")
    sensor_placement_QUBO = {node: pyo.value(model_QUBO.x[node]) for node in model_QUBO.nodes}
    return sensor_placement_MIQP, sensor_placement_QUBO


def iterate_over_rho(s, rho_values):
    """
    Iterate over different rho values for the QUBO problem and find the objective value.
    """
    approx_errors_SA = []
    approx_errors_tabu = []

    for rho in rho_values:
        Q, cQ = build_Q_matrix(G, VC, EB, s, rho)
        bqm = QUBO_dimod(Q, beta=cQ) # returns bqm value

        # Run Simulated Annealing
        print("Running Simulated Annealing...")
        simAnnSampler = neal.SimulatedAnnealingSampler()
        simAnnSamples = simAnnSampler.sample(bqm, num_reads=1000)
        simAnnSamples.info
        # Find the minimum energy sample
        min_energy_sample = min(simAnnSamples.data(), key=lambda x: x.energy)
        print("Minimum energy sample:", min_energy_sample)

        # Extract the objective value from the samples
        objective_value_SA = min_energy_sample.energy
        approx_SA = calculate_approximation_ratio(objective_value_list_MIP_min, objective_value_SA)
        approx_errors_SA.append(approx_SA)

        # Run Tabu Search
        print("Running Tabu Search...")
        tabuSampler = TabuSampler()
        tabuSamples = tabuSampler.sample(bqm, num_reads=1000)
        tabuSamples.info
        # Find the minimum energy sample
        min_energy_sample_tabu = min(tabuSamples.data(), key=lambda x: x.energy)
        print("Minimum energy sample:", min_energy_sample_tabu)

        # Extract the objective value from the samples
        objective_value_tabu = min_energy_sample_tabu.energy
        approx_Tabu = calculate_approximation_ratio(objective_value_list_MIP_min, objective_value_tabu)
        approx_errors_tabu.append(approx_Tabu)

    return approx_errors_SA, approx_errors_tabu
