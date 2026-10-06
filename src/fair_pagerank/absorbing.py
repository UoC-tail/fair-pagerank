"""Absorbing ("sink") dangling convention, for the robustness check only.

Dangling rows of T are zero: the walker is truncated there and nothing is
added to the graph. PageRank mass then sums to < 1, so the objective is the
target-group SHARE F = pi[S] / sum(pi). Gains follow the row-replacement
lemma with t_u = 0 for dangling sources.
"""

import numpy as np
import scipy.sparse as sp
from graph_cert.utils import edges_to_sparse, flip_edges, top_k_numba
from scipy.sparse.linalg import splu


def _build(adj, alpha):
    """LU of (I - alpha T0) with ZERO dangling rows."""
    adj = adj.tocsr().astype(np.float64)
    n = adj.shape[0]
    d = np.asarray(adj.sum(1)).ravel()
    inv = np.zeros(n)
    inv[d > 0] = 1.0 / d[d > 0]
    t0 = sp.diags(inv) @ adj
    return splu((sp.eye(n) - alpha * t0).tocsc()), n


def pagerank_absorbing(adj, teleport, alpha=0.85):
    lu, _ = _build(adj, alpha)
    # pi^T = (1 - alpha) teleport^T M  ->  solve the transposed system
    return (1 - alpha) * lu.solve(teleport, trans="T")


def share(adj, mask, teleport, alpha=0.85):
    pi = pagerank_absorbing(adj, teleport, alpha)
    return pi[mask].sum() / pi.sum()


def gains_absorbing(edges, adj, mask, teleport, alpha=0.85):
    """Row-replacement lemma under absorb: t_u = unif(N(u)), or 0 if dangling."""
    r = alpha / (1 - alpha)
    lu, n = _build(adj, alpha)
    adj = adj.tocsr()
    pi = (1 - alpha) * lu.solve(teleport, trans="T")
    rho = (1 - alpha) * lu.solve(mask.astype(np.float64))  # M 1_S
    g = np.zeros(len(edges))
    order = np.argsort(edges[:, 0], kind="stable")
    e_u = np.zeros(n)
    i = 0
    while i < len(order):
        u = edges[order[i], 0]
        j = i
        while j < len(order) and edges[order[j], 0] == u:
            j += 1
        e_u[u] = 1.0
        col = (1 - alpha) * lu.solve(e_u)  # p_.(u)
        e_u[u] = 0.0
        nbrs = adj.indices[adj.indptr[u] : adj.indptr[u + 1]]
        d_u = len(nbrs)
        for k in order[i:j]:
            z = edges[k, 1]
            if d_u == 0:  # t_u = 0-row: rho_t = 0, p_t(u) = 0
                g[k] = pi[u] * r * rho[z] / (1 - r * col[z])
            else:
                dr = (rho[z] - rho[nbrs].mean()) / (d_u + 1)
                dp = (col[z] - col[nbrs].mean()) / (d_u + 1)
                g[k] = pi[u] * r * dr / (1 - r * dp)
        i = j
    return g


def greedy_frec_absorbing(adj, mask, teleport, cand, rounds=10, alpha=0.85):
    """FREC+ rounds under the absorbing convention; returns the final graph."""
    adj = adj.tocsr().astype(np.float64)
    n = adj.shape[0]
    sources = np.unique(cand[:, 0])
    for _ in range(rounds):
        if len(cand) == 0:
            break
        g = gains_absorbing(cand, adj, mask, teleport, alpha)
        rows = []
        for src in sources:
            sel = cand[:, 0] == src
            if not sel.any():
                continue
            b = g[sel].argmax()
            if g[sel][b] > 0:
                rows.append(cand[sel][b])
        if not rows:
            break
        rows = np.array(rows)
        adj = flip_edges(adj, rows).tocsr()
        keys = cand[:, 0].astype(np.int64) * n + cand[:, 1]
        rm = rows[:, 0].astype(np.int64) * n + rows[:, 1]
        cand = cand[~np.isin(keys, rm)]
    return adj


def policy_iteration_absorbing(adj, mask, teleport, cand, budget=10, max_iter=50, alpha=0.85):
    """Local-budget policy iteration with the absorbing value solve; returns
    the best edge set seen (by its exact share)."""
    n = adj.shape[0]
    adj = adj.tocsr()
    reward = mask.astype(np.float64)
    lb = np.repeat(budget, n)
    ri = reward[cand[:, 0]]
    add_rem = 1 - 2 * np.asarray(adj[cand[:, 0], cand[:, 1]]).ravel()
    cur = np.array([])
    prev = np.array([[0, 0]])
    best, best_f = cur, -np.inf
    stale = 0
    for _ in range(max_iter):
        af = flip_edges(adj, cur).tocsr()
        lu, _ = _build(af, alpha)
        v = lu.solve(reward)  # (I - a T0)^-1 r
        f = share(af, mask, teleport, alpha)
        if f > best_f + 1e-9:
            best_f, best, stale = f, cur, 0
        else:
            stale += 1
        ch = (v[cand[:, 1]] - (v[cand[:, 0]] - ri) / alpha) * add_rem
        imp = ch > 0
        cur = top_k_numba(edges_to_sparse(cand[imp], n, ch[imp]), lb)
        if stale >= 6 or (edges_to_sparse(prev, n) - edges_to_sparse(cur, n)).nnz == 0:
            break
        prev = cur.copy()
    return best
