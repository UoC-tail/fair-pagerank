"""Variability across source samples at local budget 10.

Re-runs the local comparison (original, static_greedy, frec_plus,
policy_iteration) for several seeds of the 10% source sample that defines the
candidate set. Seed 0 is the sample used everywhere else. The paper reports
the standard deviation across seeds 0-4 and the per-seed difference
policy_iteration - frec_plus.

Output: results/<dataset>/seeds.csv (seed, method, F, candidates). Rows of
seeds not recomputed are kept, so seeds can run as separate jobs.
"""

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
from fair_pagerank.selection import greedy_frec, select_top_k_local


def add_arguments(parser):
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="source-sample seeds"
    )
    parser.add_argument("--budget", type=int, default=BUDGET, help="local budget (default: 10)")


def run_seed(ds, seed, budget):
    adj, mask, tel, n = ds["adj"], ds["mask"], ds["teleport"], ds["n"]
    pi = pagerank(adj, ALPHA, tel)
    cand = candidates(adj, seed)
    gains = edge_gains(cand, adj, pi, mask, ALPHA)

    def F(edges):
        return mass_after_flip(adj, edges, mask, ALPHA, tel)

    _, frec_history = greedy_frec(adj, ALPHA, mask, cand.copy(), rounds=budget)
    policy, _, _ = maximize_group_mass(adj, ALPHA, cand, local_budget(n, budget), tel, mask)
    values = {
        "original": group_mass(pi, mask),
        "static_greedy": F(select_top_k_local(cand, n, gains, budget)),
        "frec_plus": float(frec_history[-1]),
        "policy_iteration": F(policy),
    }
    return [{"seed": seed, "method": m, "F": v, "candidates": len(cand)} for m, v in values.items()]


def run(args):
    ds = setup(args)
    rows = []
    for seed in args.seeds:
        rows += run_seed(ds, seed, args.budget)
    write_csv(rows, args, "seeds", merge_on=["seed", "method"])
