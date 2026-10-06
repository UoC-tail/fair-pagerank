"""Relaxed QCLP for a global budget (graph_cert), rounded to exactly B edges."""

import cvxpy as cp
import numpy as np
from graph_cert.certify import relaxed_qclp as _relaxed_qclp
from graph_cert.certify import upper_bounds_max_ppr_target
from joblib import Parallel, delayed

from fair_pagerank.graph import minimal_dangling_repair

DEFAULT_SOLVER = "HIGHS"


def _repair(adj):
    """graph_cert's relaxed_qclp / upper bounds form D^-1 A, so dangling rows
    get one out-edge each (minimal in-oriented repair); QCLP path only."""
    d = np.asarray(adj.tocsr().sum(axis=1)).ravel()
    if (d == 0).any():
        return minimal_dangling_repair(adj)
    return adj


def compute_qclp_upper_bounds(adj, alpha, fragile, local_budget, n_jobs=-1):
    """Per-target upper bounds for the QCLP's global constraint (one
    graph_cert policy iteration per node). Targets for which no fragile edge
    improves the PPR make graph_cert crash; they get the trivially valid
    bound 1.0."""
    n = adj.shape[0]
    if not isinstance(local_budget, np.ndarray):
        local_budget = np.full(n, local_budget, dtype=int)
    adj_r = _repair(adj)

    def safe_target(t):
        try:
            return upper_bounds_max_ppr_target(adj_r, alpha, fragile, local_budget, t)
        except ValueError:
            return np.ones(n)

    results = Parallel(n_jobs=n_jobs)(delayed(safe_target)(t) for t in range(n))
    return np.column_stack(results)


def _solve_with(fn, solver, **kwargs):
    """graph_cert's relaxed_qclp hard-codes solver='GUROBI'. For any other
    solver, route its cvxpy solve to `solver` (None = cvxpy's default). The
    LP is solved exactly either way."""
    if solver is not None and solver.upper() == "GUROBI":
        return fn(**kwargs)
    original_solve = cp.problems.problem.Problem.solve

    def solve_with_solver(self, *args, **kw):
        kw.pop("solver", None)
        if solver is not None:
            kw["solver"] = solver.upper()
        return original_solve(self, **kw)

    cp.problems.problem.Problem.solve = solve_with_solver
    try:
        return fn(**kwargs)
    finally:
        cp.problems.problem.Problem.solve = original_solve


def relaxed_qclp_fairness(
    adj,
    alpha,
    fragile,
    local_budget,
    mask,
    teleport,
    global_budget=None,
    upper_bounds=None,
    solver=DEFAULT_SOLVER,
):
    """Relaxed QCLP maximizing the target-group mass under local budgets and a
    global budget, returning at most `global_budget` edges.

    graph_cert's global constraint is a weighted certification bound that
    lets the LP flip 2-3.4x more than B edges at nominal budget B. To spend
    exactly B edges, the flipped edges are ranked by the fraction of their
    auxiliary mass routed to the flipped state (ties: absolute flipped mass)
    and the top B are kept.

    Returns (edges, LP objective value).
    """
    n = adj.shape[0]
    reward = mask.astype(float)
    if not isinstance(local_budget, np.ndarray):
        local_budget = np.full(n, local_budget, dtype=int)

    adj = _repair(adj)
    if global_budget is not None and upper_bounds is None:
        upper_bounds = compute_qclp_upper_bounds(adj, alpha, fragile, local_budget)

    xval, opt_edges, obj_value, _ = _solve_with(
        _relaxed_qclp,
        solver,
        adj=adj,
        alpha=alpha,
        fragile=fragile,
        local_budget=local_budget,
        reward=reward,
        teleport=teleport,
        global_budget=global_budget,
        upper_bounds=upper_bounds,
    )

    if global_budget is not None and len(opt_edges) > global_budget:
        n_fragile = len(fragile)
        off_mass = xval[n : n + n_fragile]
        on_mass = xval[n + n_fragile :]
        exists = np.asarray(adj.tocsr()[fragile[:, 0], fragile[:, 1]]).ravel() > 0
        flip_mass = np.where(exists, off_mass, on_mass)
        keep_mass = np.where(exists, on_mass, off_mass)
        frac = flip_mass / (flip_mass + keep_mass + 1e-30)
        flipped = (on_mass > off_mass) != exists
        idx = np.where(flipped)[0]
        order = idx[np.lexsort((-flip_mass[idx], -frac[idx]))]
        chosen = order[: int(global_budget)]
        opt_edges = fragile[np.sort(chosen)]

    return opt_edges, obj_value
