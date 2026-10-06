"""Local budgets: target-group mass F after adding up to b edges per source.

For b = 0..max-budget, on the 4-hop candidates of a 10% source sample (seed 0):
  original          PageRank of the unchanged graph
  static_greedy     candidates ranked once by exact single-edge gain, top b per source
  frec_plus         round-based greedy FREC extended with the dangling-source gain
                    (FREC+; b rounds, one edge per source per round)
  policy_iteration  optimal edge set under local budget b

Output: results/<dataset>/local.csv (method, budget, F).
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
from fair_pagerank.selection import greedy_frec, select_top_k_local


def add_arguments(parser):
    parser.add_argument(
        "--max-budget", type=int, default=BUDGET, help="largest local budget (default: 10)"
    )


def run(args):
    ds = setup(args)
    adj, mask, tel, n = ds["adj"], ds["mask"], ds["teleport"], ds["n"]
    cand = candidates(adj)
    pi = pagerank(adj, ALPHA, tel)
    base = group_mass(pi, mask)
    gains = edge_gains(cand, adj, pi, mask, ALPHA)
    _, frec_history = greedy_frec(adj, ALPHA, mask, cand.copy(), rounds=args.max_budget)

    res = {m: [base] for m in ("original", "static_greedy", "frec_plus", "policy_iteration")}
    res["frec_plus"] = list(frec_history[: args.max_budget + 1])
    for b in range(1, args.max_budget + 1):
        res["original"].append(base)
        static = select_top_k_local(cand, n, gains, np.full(n, b))
        res["static_greedy"].append(mass_after_flip(adj, static, mask, ALPHA, tel))
        policy, _, _ = maximize_group_mass(adj, ALPHA, cand, local_budget(n, b), tel, mask)
        res["policy_iteration"].append(mass_after_flip(adj, policy, mask, ALPHA, tel))

    rows = [
        {"method": m, "budget": b, "F": float(v)}
        for m, vals in res.items()
        for b, v in enumerate(vals)
    ]
    write_csv(rows, args, "local")
