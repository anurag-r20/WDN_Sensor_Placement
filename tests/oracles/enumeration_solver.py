"""Pseudo-oracles for the sensor placement model: independent ways to find the optimum by brute force.

The notebook's sensor placement problem is

    min_x  sum_i c_i x_i  +  sum_(i,j) w_ij (1 - x_i)(1 - x_j)
    s.t.   sum_i x_i = s                (the documented model; MIQP() in Utils uses <= s)
           x_i in {0, 1}

EnumerationSolver tries every one of the 2^n placements and keeps the cheapest feasible one. It works
straight from the problem definition (vertex costs and edge weights), not from the Q matrix, so it shares
no code with build_Q_matrix, QUBO_dimod or the Pyomo models it is used to check. It is slow, so it only
makes sense on graphs with a dozen nodes or fewer, but each step can be checked by reading it.

enumerate_binary_pyomo_model does the same for a Pyomo model: it sets every binary variable, checks every
constraint, and evaluates the objective. It lets the tests check MIQP() and QUBO() without Gurobi.

Both live with the tests because nothing in the program uses them.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pyomo.environ as pyo

# Pyomo evaluates constraint bodies with floating-point arithmetic, so "body <= upper" is checked with
# a small allowance rather than exactly.
FEASIBILITY_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class Placement:
    """A sensor placement and its objective value.

    Attributes:
        sensors: node -> 1 if a sensor is placed there, 0 otherwise.
        objective_value: the value of the objective above, without any penalty term.
    """

    sensors: dict
    objective_value: float


def placement_cost(vertex_cost: dict, edge_weight: dict, sensors: dict) -> float:
    """Returns the objective of the sensor placement problem for one placement (no penalty term)."""
    node_term = sum(vertex_cost[node] * sensors[node] for node in vertex_cost)
    edge_term = sum(weight * (1 - sensors[i]) * (1 - sensors[j]) for (i, j), weight in edge_weight.items())
    return node_term + edge_term


class EnumerationSolver:
    """Finds the cheapest sensor placement by trying every placement."""

    def run(self, vertex_cost: dict, edge_weight: dict, s: int, relation: str = "=") -> Placement | None:
        """Returns the cheapest placement with exactly s sensors (relation "=") or at most s ("<=").

        Returns None when no placement satisfies the sensor count, e.g. s larger than the number of nodes
        with relation "=". Ties are broken by the first placement found, so tests should compare objective
        values, never the placement itself, unless the optimum is known to be unique.

        Raises:
            ValueError: if relation is neither "=" nor "<=".
        """
        if relation not in ("=", "<="):
            raise ValueError(f"relation must be '=' or '<=', got {relation!r}")
        nodes = list(vertex_cost)
        best: Placement | None = None
        for bits in itertools.product((0, 1), repeat=len(nodes)):
            count = sum(bits)
            if relation == "=" and count != s:
                continue
            if relation == "<=" and count > s:
                continue
            sensors = dict(zip(nodes, bits))
            value = placement_cost(vertex_cost, edge_weight, sensors)
            if best is None or value < best.objective_value:
                best = Placement(sensors=sensors, objective_value=value)
        return best


def enumerate_binary_pyomo_model(model: pyo.ConcreteModel) -> Placement | None:
    """Minimizes a Pyomo model over its binary variables model.x by trying every assignment.

    A placement is feasible when every active constraint holds within FEASIBILITY_TOLERANCE. Returns the
    cheapest feasible placement, or None when no placement is feasible. The model's variables are left set
    to the last assignment tried, so do not reuse their values afterwards.
    """
    nodes = list(model.nodes)
    constraints = list(model.component_data_objects(pyo.Constraint, active=True))
    best: Placement | None = None
    for bits in itertools.product((0, 1), repeat=len(nodes)):
        for node, bit in zip(nodes, bits):
            model.x[node].value = bit
        if not all(_holds(constraint) for constraint in constraints):
            continue
        value = pyo.value(model.obj)
        if best is None or value < best.objective_value:
            best = Placement(sensors=dict(zip(nodes, bits)), objective_value=value)
    return best


def _holds(constraint) -> bool:
    """True when lower <= body <= upper, within FEASIBILITY_TOLERANCE, for one Pyomo constraint."""
    body = pyo.value(constraint.body)
    lower = pyo.value(constraint.lower) if constraint.lower is not None else None
    upper = pyo.value(constraint.upper) if constraint.upper is not None else None
    if lower is not None and body < lower - FEASIBILITY_TOLERANCE:
        return False
    if upper is not None and body > upper + FEASIBILITY_TOLERANCE:
        return False
    return True
