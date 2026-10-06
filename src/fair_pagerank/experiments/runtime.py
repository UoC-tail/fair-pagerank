"""Wall time at local budget 10 on the standard candidate set.

Times, end to end: policy_iteration, static_greedy (gains + top-k selection)
and frec_plus (all rounds). Loading the graph, building the candidates and the
initial PageRank are not timed. Before timing, every method runs once,
untimed, on NBA, so that just-in-time compilation is not charged to the
measured dataset. The paper's timings ran in one job on one CPU node
(16 cores allocated).

Output: results/<dataset>/runtime.csv (method, seconds, n, edges, candidates,
sources, budget, host).
"""

import argparse
import logging
import platform
import time

import numpy as np

from fair_pagerank.datasets import load_dataset
from fair_pagerank.experiments.common import (
    ALPHA,
    BUDGET,
    candidates,
    local_budget,
    setup,
    write_csv,
)
from fair_pagerank.gain import edge_gains
from fair_pagerank.pagerank import pagerank
from fair_pagerank.policy import maximize_group_mass
from fair_pagerank.selection import greedy_frec, select_top_k_local

log = logging.getLogger(__name__)


def add_arguments(parser):
    parser.add_argument("--budget", type=int, default=BUDGET, help="local budget (default: 10)")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="untimed warm-up run on NBA (default: on)",
    )


def warmup(data_dir):
    """Run every timed method once on NBA (budget 1, one round), untimed."""
    ds = load_dataset("nba", data_dir)
    adj, mask, tel, n = ds["adj"], ds["mask"], ds["teleport"], ds["n"]
    cand = candidates(adj)
    pi = pagerank(adj, ALPHA, tel)
    maximize_group_mass(adj, ALPHA, cand, np.ones(n, dtype=int), tel, mask)
    select_top_k_local(cand, n, edge_gains(cand, adj, pi, mask, ALPHA), 1)
    greedy_frec(adj, ALPHA, mask, cand.copy(), rounds=1)
    log.info("warm-up done")


def run(args):
    if args.warmup:
        warmup(args.data_dir)
    ds = setup(args)
    adj, mask, tel, n = ds["adj"], ds["mask"], ds["teleport"], ds["n"]
    cand = candidates(adj)
    pi = pagerank(adj, ALPHA, tel)
    info = {
        "n": n,
        "edges": int(adj.nnz),
        "candidates": len(cand),
        "sources": len(np.unique(cand[:, 0])),
        "budget": args.budget,
        "host": platform.node(),
    }
    rows = []

    def record(method, t0):
        rows.append({"method": method, "seconds": round(time.perf_counter() - t0, 3), **info})
        log.info("%-18s %10.1f s", method, rows[-1]["seconds"])

    t0 = time.perf_counter()
    maximize_group_mass(adj, ALPHA, cand, local_budget(n, args.budget), tel, mask)
    record("policy_iteration", t0)

    t0 = time.perf_counter()
    gains = edge_gains(cand, adj, pi, mask, ALPHA)
    select_top_k_local(cand, n, gains, args.budget)
    record("static_greedy", t0)

    t0 = time.perf_counter()
    greedy_frec(adj, ALPHA, mask, cand.copy(), rounds=args.budget)
    record("frec_plus", t0)

    write_csv(rows, args, "runtime")
