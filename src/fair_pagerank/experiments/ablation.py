"""Contribution of each component, at local budget 10.

  original          PageRank of the unchanged graph
  original_gain     greedy FREC restricted to non-dangling sources, i.e. the reach
                    of the original FREC gain formula (undefined for d_u = 0)
  frec_plus         greedy FREC with the dangling-source gain, all candidates
  policy_iteration  policy iteration, all candidates

Output: results/<dataset>/ablation.csv (method, F, n_edges, candidates).
"""

import logging

import numpy as np

from fair_pagerank.experiments.common import (
    ALPHA,
    BUDGET,
    candidates,
    local_budget,
    setup,
    write_csv,
)
from fair_pagerank.graph import out_degree
from fair_pagerank.pagerank import group_mass, mass_after_flip, pagerank
from fair_pagerank.policy import maximize_group_mass
from fair_pagerank.selection import greedy_frec

log = logging.getLogger(__name__)


def add_arguments(parser):
    parser.add_argument("--budget", type=int, default=BUDGET, help="local budget (default: 10)")


def run(args):
    ds = setup(args)
    adj, mask, tel, n = ds["adj"], ds["mask"], ds["teleport"], ds["n"]
    cand = candidates(adj)
    non_dangling = cand[out_degree(adj)[cand[:, 0]] > 0]
    log.info(
        "sources with candidates: %d, of which non-dangling: %d",
        len(np.unique(cand[:, 0])),
        len(np.unique(non_dangling[:, 0])),
    )

    def F(edges):
        return mass_after_flip(adj, edges, mask, ALPHA, tel)

    base = group_mass(pagerank(adj, ALPHA, tel), mask)
    rows = [{"method": "original", "F": base, "n_edges": 0, "candidates": 0}]
    for label, c in [("original_gain", non_dangling), ("frec_plus", cand)]:
        edges, _ = greedy_frec(adj, ALPHA, mask, c.copy(), rounds=args.budget)
        rows.append({"method": label, "F": F(edges), "n_edges": len(edges), "candidates": len(c)})
    policy, _, _ = maximize_group_mass(adj, ALPHA, cand, local_budget(n, args.budget), tel, mask)
    rows.append(
        {
            "method": "policy_iteration",
            "F": F(policy),
            "n_edges": len(policy),
            "candidates": len(cand),
        }
    )
    write_csv(rows, args, "ablation")
