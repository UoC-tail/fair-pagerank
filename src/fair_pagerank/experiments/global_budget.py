"""Global budgets: F after adding exactly B edges in total.

The edge pools come from the local methods at local budget 10: the policy
pool (policy iteration) and the FREC+ set. On a grid of `steps` + 1 budgets
B = 0, |pool|/steps, ..., |pool|:
  original       PageRank of the unchanged graph
  greedy_gain    FREC+ set ranked once by exact single-edge gain, top B
  policy_gain    policy pool ranked once by exact single-edge gain, top B
  qclp           relaxed QCLP over the policy pool, rounded to B edges
  random_subset  B policy-pool edges drawn uniformly at random (mean of seeds 0-2)

Output: results/<dataset>/global.csv (method, budget, F).
"""

import numpy as np

from fair_pagerank.experiments.common import (
    ALPHA,
    BUDGET,
    candidates,
    local_budget,
    setup,
    write_csv,
)
from fair_pagerank.gain import edge_gains
from fair_pagerank.pagerank import group_mass, mass_after_flip, pagerank
from fair_pagerank.policy import maximize_group_mass
from fair_pagerank.qclp import DEFAULT_SOLVER, compute_qclp_upper_bounds, relaxed_qclp_fairness
from fair_pagerank.selection import greedy_frec, select_top_k_global

RANDOM_SEEDS = (0, 1, 2)


def add_qclp_arguments(parser):
    parser.add_argument(
        "--solver",
        default=DEFAULT_SOLVER,
        help="cvxpy LP solver for the QCLP (default: HIGHS; GUROBI needs the gurobi extra)",
    )
    parser.add_argument(
        "--jobs", type=int, default=-1, help="parallel jobs for the QCLP upper bounds (-1: all)"
    )


def add_arguments(parser):
    parser.add_argument(
        "--budget", type=int, default=BUDGET, help="local budget of the pools (default: 10)"
    )
    parser.add_argument("--steps", type=int, default=10, help="number of global budgets B > 0")
    add_qclp_arguments(parser)


def run(args):
    ds = setup(args)
    adj, mask, tel, n = ds["adj"], ds["mask"], ds["teleport"], ds["n"]
    cand = candidates(adj)
    pi = pagerank(adj, ALPHA, tel)
    base = group_mass(pi, mask)
    lb = local_budget(n, args.budget)

    pool, _, _ = maximize_group_mass(adj, ALPHA, cand, lb, tel, mask)
    pool_gains = edge_gains(pool, adj, pi, mask, ALPHA)
    frec_set, _ = greedy_frec(adj, ALPHA, mask, cand.copy(), rounds=args.budget)
    frec_gains = edge_gains(frec_set, adj, pi, mask, ALPHA)
    grid = [
        min(i * max(1, round(len(pool) / args.steps)), len(pool)) for i in range(args.steps + 1)
    ]
    upper_bounds = compute_qclp_upper_bounds(adj, ALPHA, pool, lb, n_jobs=args.jobs)

    def F(edges):
        return mass_after_flip(adj, edges, mask, ALPHA, tel)

    methods = ("original", "greedy_gain", "policy_gain", "qclp", "random_subset")
    res = {m: [base] for m in methods}
    for B in grid[1:]:
        res["original"].append(base)
        res["greedy_gain"].append(F(select_top_k_global(frec_set, frec_gains, B)))
        res["policy_gain"].append(F(select_top_k_global(pool, pool_gains, B)))
        qclp_edges, _ = relaxed_qclp_fairness(
            adj,
            ALPHA,
            pool,
            lb,
            mask,
            tel,
            global_budget=B,
            upper_bounds=upper_bounds,
            solver=args.solver,
        )
        res["qclp"].append(F(qclp_edges))
        rand = []
        for seed in RANDOM_SEEDS:
            pick = np.random.RandomState(seed).choice(len(pool), size=B, replace=False)
            rand.append(F(pool[pick]))
        res["random_subset"].append(float(np.mean(rand)))

    rows = [
        {"method": m, "budget": B, "F": float(v)}
        for m, vals in res.items()
        for B, v in zip(grid, vals, strict=True)
    ]
    write_csv(rows, args, "global")
