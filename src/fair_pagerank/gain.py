"""Exact single-edge fairness gains (row-replacement lemma), matrix-free."""

import numpy as np
from tqdm import tqdm

from fair_pagerank.pagerank import transition_solver


def edge_gains(edges, adj, pi, mask, alpha=0.85):
    """Exact change of the target-group mass F for adding each edge alone.

    Uses one sparse solve for rho = M 1_S (rho[v] = p_v(S)) and one per
    DISTINCT candidate source u for the PPR column M e_u (col[v] = p_v(u)),
    shared by all edges of that source: 1 + #sources solves, O(n) memory.

    Non-dangling sources (d_u >= 1) use the FREC gain of Tsioutsiouliklis et
    al. (WWW'22, Thm 4.1). DANGLING sources (d_u = 0) use the exact
    first-out-edge gain: under teleport semantics the transition row of u is
    the uniform teleport t, and adding (u, v) replaces it with e_v, giving

        gain(u, v) = pi_u * r * (rho[v] - mean_t(rho))
                     / (1 - r * (col[v] - mean_t(col))),   r = alpha/(1-alpha).

    This extension is what turns greedy FREC into FREC+. Assumes uniform
    teleport (as transition_solver does).
    """
    solve, n, adj0, _ = transition_solver(adj, alpha)

    pr_red = (1.0 - alpha) * solve(np.asarray(mask, dtype=np.float64))

    edges = np.asarray(edges)
    pi = np.ascontiguousarray(pi, dtype=np.float64)
    r = alpha / (1.0 - alpha)
    gains = np.zeros(len(edges), dtype=np.float64)
    if len(edges) == 0:
        return gains

    indptr = adj0.indptr
    indices = adj0.indices
    sources = edges[:, 0].astype(np.int64)
    order = np.argsort(sources, kind="stable")

    e_u = np.zeros(n, dtype=np.float64)
    i = 0
    m = len(order)
    with tqdm(total=m, desc="Computing gains", disable=None) as pbar:
        while i < m:
            u = sources[order[i]]
            j = i
            while j < m and sources[order[j]] == u:
                j += 1
            group = order[i:j]

            nbrs = indices[indptr[u] : indptr[u + 1]]
            d_u = len(nbrs)

            e_u[u] = 1.0
            col = (1.0 - alpha) * solve(e_u)
            e_u[u] = 0.0

            if d_u == 0:
                # dangling source: row t -> e_v (exact first-out-edge gain)
                pr_t_mean = pr_red.mean()
                ps_t_mean = col.mean()
                for idx in group:
                    v = edges[idx, 1]
                    numer = r * (pr_red[v] - pr_t_mean)
                    denom = 1.0 - r * (col[v] - ps_t_mean)
                    gains[idx] = numer / denom * pi[u]
            else:
                pr_nbr_mean = pr_red[nbrs].mean()
                ps_nbr_mean = col[nbrs].mean()
                for idx in group:
                    v = edges[idx, 1]
                    numer = r * (pr_red[v] - pr_nbr_mean)
                    denom = d_u + 1.0 - r * (col[v] - ps_nbr_mean)
                    gains[idx] = numer / denom * pi[u]

            i = j
            pbar.update(len(group))

    return gains
