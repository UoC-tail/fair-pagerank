"""Stopping at parity with the global-budget selector.

The policy pool (policy iteration at local budget 10) is ranked once by exact
single-edge gain; bisection over the number of top-ranked edges added finds
the smallest B* whose F reaches the parity target |S|/n (the target group's
share of the nodes).

Output: results/<dataset>/parity.csv with
  parity    |S|/n                  original  F of the unchanged graph
  pool      size of the policy pool F_all     F after adding the whole pool
  B_star    smallest B with F >= parity (0: already at parity; -1: unreachable)
  F_star    F at B_star (F_all if unreachable)
  residual  F_star - parity        F_prev    F at B_star - 1 (if bisected)
  crosses   whether parity is reachable within the pool
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
from fair_pagerank.gain import edge_gains
from fair_pagerank.pagerank import group_mass, mass_after_flip, pagerank
from fair_pagerank.policy import maximize_group_mass

log = logging.getLogger(__name__)


def add_arguments(parser):
    parser.add_argument(
        "--budget", type=int, default=BUDGET, help="local budget of the pool (default: 10)"
    )


def run(args):
    ds = setup(args)
    adj, mask, tel, n = ds["adj"], ds["mask"], ds["teleport"], ds["n"]
    parity = float(mask.mean())
    cand = candidates(adj)
    pi = pagerank(adj, ALPHA, tel)
    base = group_mass(pi, mask)
    pool, _, _ = maximize_group_mass(adj, ALPHA, cand, local_budget(n, args.budget), tel, mask)
    pool = np.asarray(pool)
    order = np.argsort(-edge_gains(pool, adj, pi, mask, ALPHA))

    def F(b):
        return base if b == 0 else mass_after_flip(adj, pool[order[:b]], mask, ALPHA, tel)

    f_all = F(len(pool))
    row = {"parity": parity, "original": base, "pool": len(pool), "F_all": f_all}
    if base >= parity:
        row.update(B_star=0, F_star=base, residual=base - parity, crosses=True, F_prev=None)
    elif f_all < parity:
        row.update(B_star=-1, F_star=f_all, residual=f_all - parity, crosses=False, F_prev=None)
    else:
        lo, hi = 0, len(pool)  # F(lo) < parity <= F(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if F(mid) >= parity:
                hi = mid
            else:
                lo = mid
        f_star = F(hi)
        row.update(B_star=hi, F_star=f_star, residual=f_star - parity, crosses=True, F_prev=F(lo))
    log.info(
        "parity %.4f, original %.4f -> B*=%s, F*=%.4f", parity, base, row["B_star"], row["F_star"]
    )
    write_csv([row], args, "parity")
