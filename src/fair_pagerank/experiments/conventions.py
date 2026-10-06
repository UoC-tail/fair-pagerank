"""Robustness to the dangling-node convention, at local budget 10.

Re-runs original / static_greedy / frec_plus / policy_iteration under two
alternatives to the standard teleport convention:
  self_loops  A + I: a self-loop on every node, so no node is dangling and every
              method runs through the textbook code path; F is the target mass.
  absorbing   dangling rows of T are zero (the walk is truncated, nothing is added
              to the graph); F is the target SHARE pi[S] / sum(pi).
The teleport results are those of the `local` experiment.

Output: results/<dataset>/conventions.csv (convention, method, F, n_edges).
"""

import logging

import numpy as np
from graph_cert.utils import flip_edges

from fair_pagerank import absorbing
from fair_pagerank.experiments.common import (
    ALPHA,
    BUDGET,
    candidates,
    local_budget,
    setup,
    write_csv,
)
from fair_pagerank.gain import edge_gains
from fair_pagerank.graph import add_self_loops, out_degree
from fair_pagerank.pagerank import group_mass, mass_after_flip, pagerank
from fair_pagerank.policy import maximize_group_mass
from fair_pagerank.selection import greedy_frec, select_top_k_local

log = logging.getLogger(__name__)
CONVENTIONS = ("self_loops", "absorbing")


def add_arguments(parser):
    parser.add_argument("--conventions", nargs="+", choices=CONVENTIONS, default=list(CONVENTIONS))
    parser.add_argument("--budget", type=int, default=BUDGET, help="local budget (default: 10)")


def self_loops(adj, mask, tel, budget):
    adj = add_self_loops(adj).tocsr().astype(np.float64)
    n = adj.shape[0]
    log.info("self-loops: %d edges, %d dangling", adj.nnz, (out_degree(adj) == 0).sum())
    pi = pagerank(adj, ALPHA, tel)
    cand = candidates(adj)
    gains = edge_gains(cand, adj, pi, mask, ALPHA)
    static = select_top_k_local(cand, n, gains, budget)
    frec_edges, frec_history = greedy_frec(adj, ALPHA, mask, cand.copy(), rounds=budget)
    policy, _, _ = maximize_group_mass(adj, ALPHA, cand, local_budget(n, budget), tel, mask)
    return [
        ("original", group_mass(pi, mask), 0),
        ("static_greedy", mass_after_flip(adj, static, mask, ALPHA, tel), len(static)),
        ("frec_plus", float(frec_history[-1]), len(frec_edges)),
        ("policy_iteration", mass_after_flip(adj, policy, mask, ALPHA, tel), len(policy)),
    ]


def absorbing_convention(adj, mask, tel, budget):
    n = adj.shape[0]
    cand = candidates(adj)
    gains = absorbing.gains_absorbing(cand, adj, mask, tel, ALPHA)
    static = select_top_k_local(cand, n, gains, budget)
    frec_graph = absorbing.greedy_frec_absorbing(adj, mask, tel, cand.copy(), budget, ALPHA)
    policy = absorbing.policy_iteration_absorbing(adj, mask, tel, cand, budget, alpha=ALPHA)

    def share(a):
        return float(absorbing.share(a, mask, tel, ALPHA))

    return [
        ("original", share(adj), 0),
        ("static_greedy", share(flip_edges(adj, static)), len(static)),
        ("frec_plus", share(frec_graph), int(frec_graph.nnz - adj.nnz)),
        ("policy_iteration", share(flip_edges(adj, policy)), len(policy)),
    ]


def run(args):
    ds = setup(args)
    adj, mask, tel = ds["adj"], ds["mask"], ds["teleport"]
    runners = {"self_loops": self_loops, "absorbing": absorbing_convention}
    rows = []
    for conv in args.conventions:
        for method, f, n_edges in runners[conv](adj, mask, tel, args.budget):
            rows.append({"convention": conv, "method": method, "F": f, "n_edges": n_edges})
    write_csv(rows, args, "conventions")
