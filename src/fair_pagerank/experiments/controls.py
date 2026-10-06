"""Controls at local budget 10 on the standard candidate set.

  original     PageRank of the unchanged graph
  trivial      each source adds up to 10 candidate edges to RANDOM target-group
               nodes, no optimization (seeds 0, 1, 2; the paper reports the mean)
  pa           generic recommender: preferential attachment (target in-degree)
  jaccard      generic recommender: Jaccard similarity (undirected neighborhoods)
  adamic_adar  generic recommender: Adamic-Adar (undirected neighborhoods)
The paper's "generic" column is the best of pa / jaccard / adamic_adar.

Output: results/<dataset>/controls.csv (method, seed, F, n_edges).
"""

from fair_pagerank.experiments.common import ALPHA, BUDGET, candidates, setup, write_csv
from fair_pagerank.pagerank import group_mass, mass_after_flip, pagerank
from fair_pagerank.recommenders import score_edges, top_k_per_source, trivial_edges

TRIVIAL_SEEDS = (0, 1, 2)


def add_arguments(parser):
    parser.add_argument("--budget", type=int, default=BUDGET, help="edges per source (default: 10)")


def run(args):
    ds = setup(args)
    adj, mask, tel = ds["adj"], ds["mask"], ds["teleport"]
    cand = candidates(adj)
    base = group_mass(pagerank(adj, ALPHA, tel), mask)

    rows = [{"method": "original", "seed": "", "F": base, "n_edges": 0}]
    for seed in TRIVIAL_SEEDS:
        edges = trivial_edges(cand, mask, args.budget, seed)
        f = mass_after_flip(adj, edges, mask, ALPHA, tel)
        rows.append({"method": "trivial", "seed": seed, "F": f, "n_edges": len(edges)})
    for label, scores in score_edges(adj, cand).items():
        edges = top_k_per_source(cand, scores, args.budget)
        f = mass_after_flip(adj, edges, mask, ALPHA, tel)
        rows.append({"method": label, "seed": "", "F": f, "n_edges": len(edges)})
    write_csv(rows, args, "controls")
