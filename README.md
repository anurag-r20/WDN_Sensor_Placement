# Sensor Placement in Water Distribution Networks (WDN)

This repository consists of notebooks using which, we formulate an optimization problem to identify the optimal placement of pressure sensors in a Water Distribution Network (WDN). The WDN is represented on a constrained graph $G(V,E)$, where $V$ is the set of vertices and $E$ is the set of edges.

We consider four different formulations of the optimization problem, including Mixed Integer Programming (MIP), Mixed Integer Quadratic Programming (MIQP), and Quadratic Unconstrained Binary Optimization (QUBO). The optimization problem is solved using three different solvers: Gurobi, Simulated Annealing, and D-Wave's implementation of Quantum Annealing via **[neal](https://github.com/dwavesystems/dwave-neal)**.

Additionally, we leverage D-Wave's package **[dwavebinarycsp](https://github.com/dwavesystems/dwavebinarycsp)** to translate constraint satisfaction problems into QUBOs. For Groebner basis computations, we use **[Sympy](https://www.sympy.org/)** for symbolic computation in Python and **[Networkx](https://networkx.github.io/)** for network models and graphs.
