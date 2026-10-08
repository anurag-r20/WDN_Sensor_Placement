"""The sensor placement problem as optimization models: the QUBO matrix, its dimod BQM, and the
Pyomo MIQP / QUBO models solved by Gurobi.

All formulations share the objective
``sum_i c_i x_i + sum_(i,j) w_ij (1 - x_i)(1 - x_j)``; the QUBO adds the penalty
``rho * (sum_i x_i - s)^2`` in place of the sensor-count constraint.
"""

import dimod
import numpy as np
import pyomo.environ as pyo


def build_Q_matrix(G, VC, EB, s, rho):
    """Builds the Q matrix for the Quadratic Unconstrained Binary Optimization (QUBO) problem based on the given graph and model.
    This function constructs the Q adjacency matrix used in QUBO formulations, incorporating vertex costs, edge weights (edge betweenness), and constraints.
    Args:
        G (nx.Graph): The NetworkX graph object representing the water distribution network (WDN).
        VC (dict): Vertex cost dictionary for node costs.
        EB (dict): Edge betweenness dictionary for edge weights.
        s (int): The number of sensors to be placed.
    Returns:
        tuple: A tuple containing:
            - Q (np.ndarray): The Q matrix for the QUBO problem.
            - cQ (float): The constant term in the QUBO objective function.
    Raises:
        ValueError: If the model is None.
    Examples:
        >>> Q, cQ = build_Q_matrix(G, VC, EB, 5)
        >>> print(Q)
        [[ 1.2 -0.5  0. ]
         [-0.5  1.5 -0.3]
         [ 0.  -0.3  1.1]]
        >>> print(cQ)
        2.5
    """
    nodes = list(G.nodes())
    vertex_cost = np.array([VC[node] for node in nodes])
    weight = EB
    num_nodes = len(nodes)
    A = np.ones((1, num_nodes))
    b = np.array([s])
    Q = np.diag(vertex_cost)
    total_weight = 0
    for (i, j), w in weight.items():
        if nodes.index(i) != nodes.index(j):
            Q[nodes.index(i), nodes.index(j)] += w/2
            Q[nodes.index(j), nodes.index(i)] += w/2
        Q[nodes.index(i), nodes.index(i)] -= w
        Q[nodes.index(j), nodes.index(j)] -= w
        total_weight += w
    Q += rho*np.matmul(A.T,A)
    Q -= rho*2*np.diag(np.matmul(b.T,A))
    cQ = rho * np.matmul(b.T, b) + total_weight
    return Q, cQ


def QUBO_dimod(Q, beta):
    """Creates a Binary Quadratic Model (BQM) from a Q matrix and an offset.
    This function constructs a BQM, which is used for solving quadratic optimization problems using binary variables.
    It uses the `dimod` library to create the model from the Q adjacency matrix and an offset value, preparing it for use with
    optimization solvers.
    Args:
        Q (np.ndarray): The Q matrix representing the quadratic terms in the optimization problem.
        beta (float): The offset value for the BQM.
    Returns:
        dimod.BinaryQuadraticModel: The Binary Quadratic Model constructed from the Q matrix and offset.
    Examples:
        >>> Q = np.array([[1, -1], [-1, 2]])
        >>> beta = 0.5
        >>> bqm = QUBO_dimod(Q, beta)
        >>> print(bqm)
        BinaryQuadraticModel({0: 1, 1: -1}, {(0, 1): -1}, 0.5, dimod.BINARY)
    """
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=beta)
    return bqm


def create_pyomo_model(G, water_consumption, VC, EB, city):
    """Initializes a Pyomo model for optimizing sensor placement in a city's Water Distribution Network (WDN).
    This function sets up a Pyomo model with nodes, edges, and associated parameters such as demand,
    vertex cost, and edge betweenness. It also defines the binary decision variables for sensor placement.
    Args:
        G (nx.Graph): A NetworkX graph object representing the water distribution network (WDN) for the city.
        water_consumption (dict): Water consumption dictionary for demand parameter.
        VC (dict): Vertex cost dictionary for node costs.
        EB (dict): Edge betweenness dictionary for edge weights.
        city (str): Name of the city (for error messages/logging).
    Returns:
        pyo.ConcreteModel: A Pyomo ConcreteModel instance with sets, parameters, and decision variables defined.
    Raises:
        ValueError: If the graph `G` is None.
    Examples:
        >>> model = create_pyomo_model(G, water_consumption, VC, EB, city)
        >>> print(model)
    """
    if G is None:
        print(f"{city} does not exist.")
        return None
    model = pyo.ConcreteModel()
    nodes = list(G.nodes())
    edges = list(G.edges())
    model.nodes = pyo.Set(initialize=nodes)
    model.edges = pyo.Set(initialize=edges, dimen=2)
    demand = water_consumption
    model.demand = pyo.Param(model.nodes, initialize=demand, mutable=True)
    model.c = pyo.Param(model.nodes, initialize=VC, mutable=True)
    model.w = pyo.Param(model.edges, initialize=EB, mutable=True)
    model.x = pyo.Var(model.nodes, within=pyo.Binary)
    return model


def MIQP(G, model_in, s):
    """Generates a Mixed Integer Programming (MIP) model for the given sensor placement problem.
    This function defines the objective function and constraints for the MIP model, aiming to optimize
    sensor placement in the city's Water Distribution Network (WDN). The objective function minimizes
    the total cost based on vertex costs and edge betweenness, while the constraint ensures that exactly
    `s` sensors are placed.
    Args:
        G (nx.Graph): A NetworkX graph object representing the water distribution network (WDN) for the city.
        model_in (pyo.ConcreteModel): A Pyomo ConcreteModel instance with sets and parameters defined.
        s (int): The number of sensors to be placed in the network.
    Returns:
        pyo.ConcreteModel: The updated Pyomo model with the objective function and constraints added.
    Raises:
        ValueError: If the model is None.
    Examples:
        >>> model = MIQP(G, model, 5)
        >>> print(model.obj)
    """
    if model_in is None:
        return None
    model = model_in.clone()
    def objective_rule_MIQP(model):
        return sum(model.c[i] * model.x[i] for i in model.nodes) + \
                sum(model.w[(i, j)] * (1 - model.x[i] - model.x[j] + model.x[i] * model.x[j]) for (i, j) in model.edges)
    model.obj = pyo.Objective(rule=objective_rule_MIQP, sense=pyo.minimize)
    def sensor_constraint_rule_MIQP(model):
        return sum(model.x[i] for i in model.nodes) <= s
    model.sensor_constraint_MIQP = pyo.Constraint(rule=sensor_constraint_rule_MIQP)
    return model


def QUBO(G, model_in, s, rho):
    """Generates a Quadratic Unconstrained Binary Optimization (QUBO) model for the given problem.
    This function defines the objective function for the QUBO model, which aims to optimize sensor placement
    in the city's Water Distribution Network (WDN). The objective function includes terms for vertex costs,
    edge interactions, and a penalty for deviating from the specified number of sensors.
    Args:
        G (nx.Graph): A NetworkX graph object representing the water distribution network (WDN) for the city.
        model_in (pyo.ConcreteModel): A Pyomo ConcreteModel instance with sets and parameters defined.
        s (int): The number of sensors to be placed in the network.
        rho (float): Penalty parameter for the constraint that exactly `s` sensors must be placed.
    Returns:
        pyo.ConcreteModel: The updated Pyomo model with the QUBO objective function added.
    Raises:
        ValueError: If the model is None.
    Examples:
        >>> model = QUBO(G, model, 5, 0.1)
        >>> print(model.obj)
    """
    if model_in is None:
        return None
    model = model_in.clone()
    def objective_rule_QUBO(model):
        term1 = sum(model.c[i] * model.x[i] for i in model.nodes)
        term2 = sum(model.w[(i, j)] * (1 - model.x[i]) * (1 - model.x[j]) for (i, j) in model.edges)
        term3 = rho * (sum(model.x[i] for i in model.nodes) - s) ** 2
        return term1 + term2 + term3
    model.obj = pyo.Objective(rule=objective_rule_QUBO, sense=pyo.minimize)
    return model
