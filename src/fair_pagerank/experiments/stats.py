"""Per-dataset statistics (the dataset table of the appendix).

  nodes, edges     size of the loaded graph (weak LCC, directed edges)
  dangling         nodes without out-links
  minority_share   share of the minority (smaller) sensitive group
  target           group whose mass is maximized (minority or majority)
  target_share     its share |S|/n: the parity target
  sources          sampled sources that have candidates
  candidates       size of the 4-hop candidate set (seed 0)
  pool             edges selected by policy iteration at local budget 10
  iterations       policy-iteration iterations until convergence

Output: results/<dataset>/stats.csv (one row).
"""

from fair_pagerank.experiments.common import (
    ALPHA,
    BUDGET,
    candidates,
    local_budget,
    setup,
    write_csv,
)
from fair_pagerank.graph import out_degree
from fair_pagerank.policy import maximize_group_mass


def add_arguments(parser):
    parser.add_argument(
        "--budget", type=int, default=BUDGET, help="local budget of the pool (default: 10)"
    )


def run(args):
    ds = setup(args)
    adj, mask, tel, n = ds["adj"], ds["mask"], ds["teleport"], ds["n"]
    cand = candidates(adj)
    pool, _, iterations = maximize_group_mass(
        adj, ALPHA, cand, local_budget(n, args.budget), tel, mask
    )
    row = {
        "nodes": n,
        "edges": int(adj.nnz),
        "dangling": int((out_degree(adj) == 0).sum()),
        "minority_share": float(ds["minority"].mean()),
        "target": ds["target"],
        "target_share": float(mask.mean()),
        "sources": len(set(cand[:, 0].tolist())),
        "candidates": len(cand),
        "pool": len(pool),
        "iterations": iterations,
    }
    write_csv([row], args, "stats")
