"""Exact optima at small global budgets B = 1..max-b, and the heuristics' gaps.

  original               F of the unchanged graph (B = 0)
  exact_unrestricted     exhaustive search over ALL candidates for B = 1 (and
                         B = 2 with --full-b2), else over the top-k candidates
                         by single-edge gain (set = "top<k>")
  exact_pool             exhaustive search over the policy pool (B <= --pool-max-b)
  rank_once_full         top-B candidates by single-edge gain
  rank_once_pool         top-B pool edges by single-edge gain ("policy gain")
  greedy_recompute_full  B sequential additions, gains recomputed after each
  qclp_pool              relaxed QCLP over the pool, rounded to B edges

Output: results/<dataset>/exact.csv (B, method, F, set).
"""

import itertools
import logging
import multiprocessing as mp
import os

import numpy as np
from graph_cert.utils import flip_edges

from fair_pagerank.experiments.common import (
    ALPHA,
    BUDGET,
    candidates,
    local_budget,
    setup,
    write_csv,
)
from fair_pagerank.experiments.global_budget import add_qclp_arguments
from fair_pagerank.gain import edge_gains
from fair_pagerank.pagerank import group_mass, mass_after_flip, pagerank
from fair_pagerank.policy import maximize_group_mass
from fair_pagerank.qclp import compute_qclp_upper_bounds, relaxed_qclp_fairness

log = logging.getLogger(__name__)
G = {}  # graph state shared with forked worker processes


def _available_cpus():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def add_arguments(parser):
    parser.add_argument("--max-b", type=int, default=3, help="largest global budget B")
    parser.add_argument("--top-k", type=int, default=200, help="reduced set size for B >= 2")
    parser.add_argument(
        "--full-b2", action="store_true", help="exhaustive B = 2 over ALL candidates"
    )
    parser.add_argument(
        "--pool-max-b", type=int, default=99, help="skip exhaustive pool search above this B"
    )
    parser.add_argument(
        "--procs", type=int, default=_available_cpus(), help="worker processes (default: all)"
    )
    parser.add_argument(
        "--budget", type=int, default=BUDGET, help="local budget of the pool (default: 10)"
    )
    add_qclp_arguments(parser)


def F_of(edges):
    return mass_after_flip(G["adj"], np.asarray(edges, dtype=int), G["mask"], ALPHA, G["tel"])


def _worker(arg):
    """Best size-B subset among the combinations assigned to this worker."""
    start, step, B, idx = arg
    cand = G["cand"][idx] if idx is not None else G["cand"]
    m = len(cand)
    best_f, best = -np.inf, None
    for i in range(start, m, step):
        if B == 1:
            combos = [(i,)]
        elif B == 2:
            combos = ((i, j) for j in range(i + 1, m))
        else:
            combos = ((i,) + c for c in itertools.combinations(range(i + 1, m), B - 1))
        for c in combos:
            f = F_of(cand[list(c)])
            if f > best_f:
                best_f, best = f, c
    return best_f, best


def exhaustive(B, procs, idx=None):
    tasks = [(w, procs, B, idx) for w in range(procs)]
    with mp.get_context("fork").Pool(procs) as p:
        res = p.map(_worker, tasks)
    return max(res, key=lambda r: r[0])[0]


def greedy_recompute(B):
    """Sequential greedy, recomputing exact single-edge gains after each add."""
    cand = G["cand"]
    cur, chosen = G["adj"].copy(), []
    avail = np.ones(len(cand), bool)
    for _ in range(B):
        pi = pagerank(cur, ALPHA, G["tel"])
        gains = edge_gains(cand, cur, pi, G["mask"], ALPHA)
        gains[~avail] = -np.inf
        k = int(np.argmax(gains))
        avail[k] = False
        chosen.append(cand[k])
        cur = flip_edges(cur, np.array([cand[k]])).tocsr()
    return F_of(np.array(chosen))


def run(args):
    ds = setup(args)
    adj, mask, tel, n = ds["adj"], ds["mask"], ds["teleport"], ds["n"]
    G.update(adj=adj, mask=mask, tel=tel, cand=candidates(adj))
    cand = G["cand"]
    pi = pagerank(adj, ALPHA, tel)
    base = group_mass(pi, mask)
    gains = edge_gains(cand, adj, pi, mask, ALPHA)
    lb = local_budget(n, args.budget)
    pool, _, _ = maximize_group_mass(adj, ALPHA, cand, lb, tel, mask)
    pool = np.asarray(pool)

    keys = cand[:, 0].astype(np.int64) * n + cand[:, 1]
    pool_keys = pool[:, 0].astype(np.int64) * n + pool[:, 1]
    pool_idx = np.where(np.isin(keys, pool_keys))[0]
    topk_idx = np.argsort(-gains)[: args.top_k]
    log.info("candidates=%d pool=%d procs=%d", len(cand), len(pool_idx), args.procs)

    rows = [{"B": 0, "method": "original", "F": base, "set": "-"}]
    for B in range(1, args.max_b + 1):
        full = B == 1 or (B == 2 and args.full_b2)
        scope = "full" if full else f"top{args.top_k}"
        exact_all = exhaustive(B, args.procs, None if full else topk_idx)
        exact_pool = exhaustive(B, args.procs, pool_idx) if args.pool_max_b >= B else np.nan
        values = [
            ("exact_unrestricted", exact_all, scope),
            ("exact_pool", exact_pool, "pool"),
            ("rank_once_full", F_of(cand[np.argsort(-gains)[:B]]), "full"),
            ("rank_once_pool", F_of(cand[pool_idx[np.argsort(-gains[pool_idx])[:B]]]), "pool"),
            ("greedy_recompute_full", greedy_recompute(B), "full"),
        ]
        for method, f, s in values:
            rows.append({"B": B, "method": method, "F": f, "set": s})
            log.info(
                "B=%d %-22s F=%.6f (gap to exact %+.4f pts)", B, method, f, 100 * (f - exact_all)
            )

    upper_bounds = compute_qclp_upper_bounds(adj, ALPHA, pool, lb, n_jobs=args.jobs)
    for B in range(1, args.max_b + 1):
        edges, _ = relaxed_qclp_fairness(
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
        rows.append({"B": B, "method": "qclp_pool", "F": F_of(edges), "set": "pool"})
    write_csv(rows, args, "exact")
