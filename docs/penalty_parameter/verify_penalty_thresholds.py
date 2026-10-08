#!/usr/bin/env python3
"""Numerical companion to penalty_thresholds.tex.

Checks every result of the note by exhaustive enumeration, and computes the per-network quantities
reported in its tables:

  1. Lemma 2 (one-flip changes) and Lemma 3 (W_i <= 2(n-1)/n for normalized edge betweenness).
  2. Theorems 4-5 and Corollary 6 (equality budget), Theorems 8-9 and Corollary 10 (inequality
     budget with slack bits): above the threshold every minimizer of the penalized QUBO is feasible
     and optimal; at the threshold the optimal value is still exact.
  3. Propositions 7 and 11 (sharpness): every instance attains its threshold at s in {0, n}, and
     below the threshold the families of 7(b), 7(c) and 11(b) have only infeasible minimizers.
  4. Proposition 15 (critical penalty from the coverage curve), against a direct check of the
     definition on either side of rho_star, and Proposition 16 (rho_star = 0 at the notebook's s*).
  5. Corollary 13 and Tables 1-2: thresholds for every network in Data/WDN_Data, and the exact
     rho_star for the networks small enough to enumerate (n <= 24).

Instances are random connected graphs scored two ways: with the repository's `centrality` (vertex
costs from demands, edge betweenness weights), and with generic costs c_i ~ U[-1, 2] and weights
w_ij ~ U[0, 2], since the theorems hold for any real c and nonnegative w.

Run from the repository root (needs numpy, networkx, pandas and the repository's src package):

    python docs/penalty_parameter/verify_penalty_thresholds.py

It writes penalty_thresholds_results.json next to this file and exits with status 1 if any check fails.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.formulations import build_Q_matrix  # noqa: E402
from src.network import Construct_Graph, WDN_network_data, centrality  # noqa: E402

SEED = 20261007
TOL = 1e-9
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


# =============================================================================
# The model, written from its definition (independent of build_Q_matrix)
# =============================================================================


def objective_table(c: np.ndarray, edges: np.ndarray, w: np.ndarray, n: int, chunk_bits: int = 18):
    """Yields (counts, f) for every x in {0,1}^n in chunks: counts = sum(x), f = objective of x."""
    total = 1 << n
    chunk = min(total, 1 << chunk_bits)
    shifts = np.arange(n, dtype=np.int64)
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        bits = ((idx[:, None] >> shifts) & 1).astype(np.float64)
        f = bits @ c
        if len(w):
            uncovered = (1.0 - bits[:, edges[:, 0]]) * (1.0 - bits[:, edges[:, 1]])
            f += uncovered @ w
        yield bits.sum(axis=1).astype(np.int64), f, bits


def coverage_curve(c, edges, w, n) -> np.ndarray:
    """m_k = min { f(x) : sum(x) = k } for k = 0..n, by exhaustive enumeration."""
    m = np.full(n + 1, np.inf)
    for counts, f, _ in objective_table(c, edges, w, n):
        np.minimum.at(m, counts, f)
    return m


def incident_weight(edges, w, n) -> np.ndarray:
    W = np.zeros(n)
    np.add.at(W, edges[:, 0], w)
    np.add.at(W, edges[:, 1], w)
    return W


def rho_bar_eq(c, W) -> float:
    """Theorem 4: max_i max(c_i, W_i - c_i)."""
    return float(np.max(np.maximum(c, W - c))) if len(c) else 0.0


def rho_bar_le(c, W) -> float:
    """Theorem 8: max(0, max_i (W_i - c_i))."""
    return float(max(0.0, np.max(W - c))) if len(c) else 0.0


def rho_bar_eq_budget(c, W, s) -> float:
    """Corollary 6: max(0, a_[s+1], c_[n-s+1]) with a = W - c and [k] the k-th largest."""
    n = len(c)
    a_sorted = np.sort(W - c)[::-1]
    c_sorted = np.sort(c)[::-1]
    terms = [0.0]
    if s < n:
        terms.append(a_sorted[s])  # (s+1)-th largest
    if s > 0:
        terms.append(c_sorted[n - s])  # (n-s+1)-th largest
    return float(max(terms))


def rho_bar_le_budget(c, W, s) -> float:
    """Corollary 10: max(0, a_[s+1]) with a = W - c."""
    a_sorted = np.sort(W - c)[::-1]
    return float(max(0.0, a_sorted[s])) if s < len(c) else 0.0


def first_coverage_minimizer(m: np.ndarray) -> int:
    """The notebook's budget rule: the first s at which the <= coverage curve M reaches its minimum."""
    M = np.minimum.accumulate(m)
    return int(np.flatnonzero(M <= M.min() + 1e-12)[0])


def rho_star_eq(m: np.ndarray, s: int) -> float:
    """Proposition 15(a): max(0, max_{k != s} (m_s - m_k) / (k - s)^2)."""
    values = [(m[s] - m[k]) / (k - s) ** 2 for k in range(len(m)) if k != s]
    return float(max([0.0] + values))


def rho_star_le(m: np.ndarray, s: int) -> float:
    """Proposition 15(b): max(0, max_{k > s} (M_s - M_k) / (k - s)^2), M the running minimum of m."""
    M = np.minimum.accumulate(m)
    values = [(M[s] - M[k]) / (k - s) ** 2 for k in range(s + 1, len(m))]
    return float(max([0.0] + values))


def slack_coefficients(s: int) -> list[int]:
    """Bounded binary encoding: nonnegative integers whose subset sums are exactly {0, ..., s}."""
    if s == 0:
        return []
    K = s.bit_length()
    return [2**k for k in range(K - 1)] + [s - (2 ** (K - 1) - 1)]


# =============================================================================
# Penalized problems, minimized by brute force
# =============================================================================


def penalized_equality(c, edges, w, n, s, rho):
    """Returns (min F, all minimizers' sensor counts) for F = f(x) + rho (sum x - s)^2."""
    best, counts_at_best = np.inf, set()
    for counts, f, _ in objective_table(c, edges, w, n):
        F = f + rho * (counts - s) ** 2
        lo = F.min()
        if lo < best - TOL:
            best, counts_at_best = lo, set()
        if lo <= best + TOL:
            counts_at_best |= set(counts[F <= best + TOL].tolist())
    return best, counts_at_best


def penalized_inequality(c, edges, w, n, s, rho):
    """Returns (min F, set of (sum x, d) over minimizers) for F = f(x) + rho (sum x + a.y - s)^2."""
    a = np.array(slack_coefficients(s), dtype=np.int64)
    slack_sums = np.array([int(np.dot(a, y)) for y in itertools.product((0, 1), repeat=len(a))], dtype=np.int64)
    best, at_best = np.inf, set()
    for counts, f, _ in objective_table(c, edges, w, n):
        d = counts[:, None] + slack_sums[None, :] - s  # every (x, y) pair
        F = f[:, None] + rho * d**2
        lo = F.min()
        if lo < best - TOL:
            best, at_best = lo, set()
        if lo <= best + TOL:
            rows, cols = np.nonzero(F <= best + TOL)
            at_best |= {(int(counts[r]), int(d[r, k])) for r, k in zip(rows, cols)}
    return best, at_best


# =============================================================================
# Random instances
# =============================================================================


def random_instance(rng: random.Random, kind: str, n_min=2, n_max=9):
    n = rng.randint(n_min, n_max)
    while True:
        G = nx.gnp_random_graph(n, rng.uniform(0.3, 0.8), seed=rng.randrange(10**9))
        if nx.is_connected(G):
            break
    G = nx.relabel_nodes(G, {i: f"J-{i}" for i in G})
    nodes = list(G.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    if kind == "centrality":
        VC, EB = centrality(G, {v: rng.uniform(1.0, 100.0) for v in nodes})
        c = np.array([VC[v] for v in nodes])
        edge_list = list(EB.items())
    else:
        c = np.array([rng.uniform(-1.0, 2.0) for _ in nodes])
        edge_list = [((u, v), rng.uniform(0.0, 2.0)) for u, v in G.edges()]
    edges = np.array([(pos[u], pos[v]) for (u, v), _ in edge_list], dtype=np.int64).reshape(-1, 2)
    w = np.array([x for _, x in edge_list])
    return G, nodes, c, edges, w


def check_lemma_one_flip(rng: random.Random, trials=200) -> None:
    for _ in range(trials):
        G, nodes, c, edges, w = random_instance(rng, rng.choice(["centrality", "generic"]))
        n = len(nodes)
        W = incident_weight(edges, w, n)
        x = np.array([rng.randint(0, 1) for _ in range(n)], dtype=float)
        i = rng.randrange(n)

        def f(v):
            return float(v @ c + np.sum(w * (1 - v[edges[:, 0]]) * (1 - v[edges[:, 1]])))

        y = x.copy()
        y[i] = 1 - y[i]
        delta = f(y) - f(x)
        uncovered = sum(wt for (a, b), wt in zip(edges, w) if i in (a, b) and x[b if a == i else a] == 0)
        expected = c[i] - uncovered if x[i] == 0 else -c[i] + uncovered
        check(abs(delta - expected) < 1e-9, f"Lemma 2 formula: {delta} vs {expected}")
        if x[i] == 0:
            check(c[i] - W[i] - 1e-9 <= delta <= c[i] + 1e-9, "Lemma 2 add bounds")
        else:
            check(-c[i] - 1e-9 <= delta <= W[i] - c[i] + 1e-9, "Lemma 2 remove bounds")


def check_lemma_edge_betweenness(rng: random.Random, trials=300) -> float:
    worst_ratio = 0.0
    for _ in range(trials):
        n = rng.randint(2, 40)
        while True:
            G = nx.gnp_random_graph(n, rng.uniform(0.05, 0.9), seed=rng.randrange(10**9))
            if nx.is_connected(G):
                break
        EB = nx.edge_betweenness_centrality(G)
        W = {v: 0.0 for v in G}
        for (u, v), x in EB.items():
            W[u] += x
            W[v] += x
        bound = 2 * (n - 1) / n
        check(max(W.values()) <= bound + 1e-12, f"Lemma 3 violated, n={n}")
        worst_ratio = max(worst_ratio, max(W.values()) / bound)
    # The bound is attained by the star's centre: every pair either ends at the centre or passes through it.
    for n in range(3, 30):
        EB = nx.edge_betweenness_centrality(nx.star_graph(n - 1))
        check(abs(sum(EB.values()) - 2 * (n - 1) / n) < 1e-12, f"Lemma 3 not attained by star, n={n}")
    return worst_ratio


def check_theorems_on_random_instances(rng: random.Random, trials=300) -> dict:
    stats = {"instances": 0, "equality_checks": 0, "inequality_checks": 0, "rho_star_checks": 0,
             "max_rho_star_over_rho_bar_eq": 0.0}
    for _ in range(trials):
        kind = rng.choice(["centrality", "generic"])
        G, nodes, c, edges, w = random_instance(rng, kind)
        n = len(nodes)
        W = incident_weight(edges, w, n)
        m = coverage_curve(c, edges, w, n)
        M = np.minimum.accumulate(m)
        s = rng.randint(0, n)
        stats["instances"] += 1

        # Theorem 4, Corollary 6 and Theorem 5 (equality budget)
        for rho_bar in (rho_bar_eq(c, W), rho_bar_eq_budget(c, W, s)):
            best, counts = penalized_equality(c, edges, w, n, s, rho_bar * (1 + 1e-6) + 1e-6)
            check(counts == {s} and abs(best - m[s]) < 1e-7, f"equality theorem fails ({kind}, n={n}, s={s})")
            best, _ = penalized_equality(c, edges, w, n, s, rho_bar)
            check(abs(best - m[s]) < 1e-7, f"equality boundary value wrong ({kind}, n={n}, s={s})")
            stats["equality_checks"] += 1
        check(rho_bar_eq_budget(c, W, s) <= rho_bar_eq(c, W) + 1e-12, "Corollary 6 not below Theorem 4")

        # Theorem 8, Corollary 10 and Theorem 9 (inequality budget with slack bits)
        rb = rho_bar_le(c, W)
        for rho_bar in (rb, rho_bar_le_budget(c, W, s)):
            best, pairs = penalized_inequality(c, edges, w, n, s, rho_bar * (1 + 1e-6) + 1e-6)
            check(all(d == 0 and k <= s for k, d in pairs) and abs(best - M[s]) < 1e-7,
                  f"inequality theorem fails ({kind}, n={n}, s={s})")
        if rb > 0:
            best, _ = penalized_inequality(c, edges, w, n, s, rb)
            check(abs(best - M[s]) < 1e-7, f"inequality boundary value wrong ({kind}, n={n}, s={s})")
        stats["inequality_checks"] += 1

        # Proposition 15: the critical penalties computed from the coverage curve
        for relation, rho_star, solve, feasible in (
            ("=", rho_star_eq(m, s), lambda r: penalized_equality(c, edges, w, n, s, r),
             lambda res: res[1] == {s}),
            ("<=", rho_star_le(m, s), lambda r: penalized_inequality(c, edges, w, n, s, r),
             lambda res: all(d == 0 and k <= s for k, d in res[1])),
        ):
            delta = 1e-6 * (1 + rho_star)
            check(feasible(solve(rho_star + delta)), f"rho_star{relation} too small ({kind}, n={n}, s={s})")
            if rho_star > 2 * delta:
                check(not feasible(solve(rho_star - delta)), f"rho_star{relation} too large ({kind}, n={n}, s={s})")
            stats["rho_star_checks"] += 1
        check(rho_star_eq(m, s) <= rho_bar_eq_budget(c, W, s) + 1e-9, "rho_star_eq above Corollary 6")
        check(rho_star_le(m, s) <= rho_bar_le_budget(c, W, s) + 1e-9, "rho_star_le above Corollary 10")

        # Propositions 7(a) and 11(a): the thresholds are attained at the extreme budgets
        a = W - c
        check(abs(rho_star_eq(m, 0) - max(0.0, a.max())) < 1e-9, "Prop 7(a): rho_star_eq(0)")
        check(abs(rho_star_eq(m, n) - max(0.0, c.max())) < 1e-9, "Prop 7(a): rho_star_eq(n)")
        check(abs(max(rho_star_eq(m, k) for k in range(n + 1)) - rho_bar_eq(c, W)) < 1e-9,
              "Prop 7(a): max_s rho_star_eq != rho_bar_eq")
        check(abs(rho_star_le(m, 0) - rb) < 1e-9, "Prop 11(a): rho_star_le(0) != rho_bar_le")
        check(abs(max(rho_star_le(m, k) for k in range(n + 1)) - rb) < 1e-9, "Prop 11(a): max_s rho_star_le")

        # Proposition 16: at the budget the notebook selects, both critical penalties vanish
        s_star = first_coverage_minimizer(m)
        check(rho_star_eq(m, s_star) == 0.0 and rho_star_le(m, s_star) == 0.0, "Prop 16 fails")
        best, counts = penalized_equality(c, edges, w, n, s_star, 1e-6)
        check(counts == {s_star}, "Prop 16: tiny rho not exact at s_star")
        if rho_bar_eq(c, W) > 0:
            stats["max_rho_star_over_rho_bar_eq"] = max(stats["max_rho_star_over_rho_bar_eq"],
                                                       rho_star_eq(m, s) / rho_bar_eq(c, W))
    return stats


def check_sharpness() -> None:
    """Propositions 7(b), 7(c) and 11(b): s + 1 disjoint pipes of weight w (costs 0); s + 1 isolated nodes of cost c."""
    for s in range(0, 4):
        n = 2 * (s + 1)
        c = np.zeros(n)
        edges = np.array([(2 * t, 2 * t + 1) for t in range(s + 1)], dtype=np.int64)
        w = np.ones(s + 1)
        W = incident_weight(edges, w, n)
        check(abs(rho_bar_eq(c, W) - 1) < 1e-12 and abs(rho_bar_le(c, W) - 1) < 1e-12, "pipes: rho_bar != w")
        _, counts = penalized_equality(c, edges, w, n, s, 0.99)
        check(counts == {s + 1}, f"pipes (=): minimizers not all infeasible at rho < w, s={s}")
        _, pairs = penalized_inequality(c, edges, w, n, s, 0.99)
        check(all(k == s + 1 for k, _ in pairs), f"pipes (<=): minimizers not all infeasible at rho < w, s={s}")
        _, counts = penalized_equality(c, edges, w, n, s, 1.0)
        check(s + 1 in counts and s in counts, f"pipes (=): no infeasible tie at rho = w, s={s}")
    for s in range(1, 5):
        n = s + 1
        c = np.ones(n)
        edges = np.zeros((0, 2), dtype=np.int64)
        w = np.zeros(0)
        W = incident_weight(edges, w, n)
        check(abs(rho_bar_eq(c, W) - 1) < 1e-12 and abs(rho_bar_eq_budget(c, W, s) - 1) < 1e-12,
              "isolated: thresholds != c")
        _, counts = penalized_equality(c, edges, w, n, s, 0.99)
        check(counts == {s - 1}, f"isolated (=): minimizers not all infeasible at rho < c, s={s}")


# =============================================================================
# The networks in Data/WDN_Data
# =============================================================================

LOGGED_BUDGET = {"Apulia": 11, "Fossolo": 15, "ZJ": 15, "Modena": 87}


def network_tables() -> list[dict]:
    rows = []
    for city in ["Test", "Apulia", "Fossolo", "ZJ", "Modena", "Kentucky"]:
        with contextlib.redirect_stdout(io.StringIO()):
            coords, edge_set, consumption = WDN_network_data(city)
        G = Construct_Graph(city, coords, edge_set)
        VC, EB = centrality(G, consumption)
        nodes = list(G.nodes())
        pos = {v: i for i, v in enumerate(nodes)}
        n = len(nodes)
        c = np.array([VC[v] for v in nodes])
        edges = np.array([(pos[u], pos[v]) for (u, v) in EB], dtype=np.int64)
        w = np.array(list(EB.values()))
        W = incident_weight(edges, w, n)
        row = {
            "network": city, "n": n, "edges": int(len(w)),
            "max_c": float(c.max()), "max_W": float(W.max()), "lemma3_bound": 2 * (n - 1) / n,
            "rho_notebook": float(np.sum(np.abs(c)) + 1),
            "rho_bar_eq": rho_bar_eq(c, W), "rho_bar_le": rho_bar_le(c, W),
            "logged_s": LOGGED_BUDGET.get(city),
        }
        check(row["max_W"] < 2 * (n - 1) / n + 1e-12, f"Lemma 3 fails on {city}")
        check(row["rho_bar_eq"] < 2, f"Corollary 13 fails on {city}")
        # Cross-check the objective against build_Q_matrix at rho = 0 on a few random placements.
        Q, cQ = build_Q_matrix(G, VC, EB, 0, 0.0)
        for x in np.random.default_rng(SEED).integers(0, 2, size=(20, n)).astype(float):
            direct = float(x @ c + np.sum(w * (1 - x[edges[:, 0]]) * (1 - x[edges[:, 1]])))
            check(abs(direct - float(x @ Q @ x + cQ)) < 1e-9, f"objective mismatch with build_Q_matrix on {city}")
        if n <= 24:
            m = coverage_curve(c, edges, w, n)
            eq = [rho_star_eq(m, s) for s in range(n + 1)]
            le = [rho_star_le(m, s) for s in range(n + 1)]
            budget = [rho_bar_eq_budget(c, W, s) for s in range(n + 1)]
            row.update({
                "coverage_curve": m.tolist(),
                "rho_star_eq_by_s": eq, "rho_star_le_by_s": le, "rho_bar_eq_budget_by_s": budget,
                "rho_star_eq_max": max(eq), "rho_star_le_max": max(le),
                "rho_star_eq_median": float(np.median(eq)), "rho_star_le_median": float(np.median(le)),
            })
            s_star = first_coverage_minimizer(m)
            row.update({"s_star": s_star, "rho_star_eq_at_s_star": eq[s_star],
                        "rho_star_le_at_s_star": le[s_star]})
            if row["logged_s"] is not None:
                s = row["logged_s"]
                row.update({"rho_star_eq_at_logged_s": eq[s], "rho_star_le_at_logged_s": le[s],
                            "rho_bar_eq_budget_at_logged_s": budget[s]})
        rows.append(row)
    return rows


def main() -> int:
    rng = random.Random(SEED)
    check_lemma_one_flip(rng)
    worst_ratio = check_lemma_edge_betweenness(rng)
    stats = check_theorems_on_random_instances(rng)
    check_sharpness()
    rows = network_tables()

    results = {"seed": SEED, "random_instance_checks": stats, "lemma3_worst_ratio_random": worst_ratio,
               "networks": rows, "failures": FAILURES}
    out = Path(__file__).with_name("penalty_thresholds_results.json")
    out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"random instances: {stats}")
    print(f"Lemma 3: worst max_i W_i / (2(n-1)/n) on random graphs = {worst_ratio:.4f} (star attains 1)")
    header = f"{'network':9s} {'n':>4s} {'max c':>7s} {'max W':>7s} {'rho_nb':>8s} {'bar_eq':>7s} {'bar_le':>7s}"
    print(header + f" {'*eq max':>8s} {'*le max':>8s} {'s_log':>5s} {'*eq(s)':>7s} {'*le(s)':>7s}")
    for r in rows:
        extra = ""
        if "rho_star_eq_max" in r:
            extra = f" {r['rho_star_eq_max']:8.4f} {r['rho_star_le_max']:8.4f}"
            if r["logged_s"] is not None:
                extra += f" {r['logged_s']:5d} {r['rho_star_eq_at_logged_s']:7.4f} {r['rho_star_le_at_logged_s']:7.4f}"
        print(f"{r['network']:9s} {r['n']:4d} {r['max_c']:7.4f} {r['max_W']:7.4f} {r['rho_notebook']:8.3f} "
              f"{r['rho_bar_eq']:7.4f} {r['rho_bar_le']:7.4f}" + extra)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    if FAILURES:
        print(f"{len(FAILURES)} CHECKS FAILED:\n  " + "\n  ".join(FAILURES[:20]))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
