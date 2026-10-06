"""PageRank with standard teleport handling of dangling nodes.

A dangling node (zero out-degree) redistributes its mass to the teleport
distribution. This is done implicitly (rank-1 term), so no dense dangling row
is ever materialized and the graph keeps its true edge count.
"""

import numpy as np
import scipy.sparse as sp
from graph_cert.utils import flip_edges
from scipy.sparse.linalg import LinearOperator, gmres, splu


def pagerank(adj, alpha=0.85, teleport=None):
    """Solve (I - alpha P) pi = (1-alpha) teleport, with P x = A^T (inv_d * x)
    + (sum of x over dangling nodes) * teleport."""
    n = adj.shape[0]
    if teleport is None:
        teleport = np.ones(n) / n

    adj = adj.tocsr().astype(np.float64)
    d = np.asarray(adj.sum(axis=1)).ravel()
    dangling = d == 0
    inv_d = np.zeros(n)
    inv_d[~dangling] = 1.0 / d[~dangling]
    adj_t = adj.T.tocsr()

    def matvec(x):
        px = adj_t @ (inv_d * x) + x[dangling].sum() * teleport
        return x - alpha * px

    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    pi, _ = gmres(op, (1 - alpha) * teleport, rtol=1e-9, atol=0.0)
    return pi


def group_mass(pi, mask):
    """F: total PageRank mass of the target group."""
    return float(pi[mask].sum())


def mass_after_flip(adj, edges, mask, alpha=0.85, teleport=None):
    """F of the graph obtained by flipping (here: adding) `edges`."""
    adj_new = flip_edges(adj, edges) if len(edges) > 0 else adj
    return group_mass(pagerank(adj_new, alpha, teleport), mask)


def transition_solver(adj, alpha=0.85):
    """Factorize (I - alpha * T) for matrix-free personalized-PageRank solves.

    Every zero-out-degree row of T equals the uniform teleport ones/n. This is
    reproduced without materializing any dense row via a rank-1
    Sherman-Morrison correction on top of the sparse LU of B = I - alpha * T0,
    where T0 has all-zero dangling rows. Only nnz == 0 rows count as dangling.

    Returns (solve, n, adj0, deg) where
      solve(b) == (I - alpha * T)^{-1} @ b,
      adj0     == the CSR adjacency (dangling rows zeroed), for neighbor lookups,
      deg      == out-degree of each node (0 for dangling).
    """
    adj = adj.tocsr().astype(np.float64)
    n = adj.shape[0]

    nnz_row = np.diff(adj.indptr)
    dangling = nnz_row == 0

    if dangling.any():
        keep = (~dangling).astype(np.float64)
        adj0 = (sp.diags(keep) @ adj).tocsr()
    else:
        adj0 = adj

    deg = np.asarray(adj0.sum(axis=1)).ravel()
    d_safe = deg.copy()
    d_safe[dangling] = 1.0

    t0 = sp.diags(1.0 / d_safe) @ adj0
    lu = splu((sp.eye(n) - alpha * t0).tocsc())

    if dangling.any():
        c = alpha / n
        w = lu.solve(dangling.astype(np.float64))
        denom_corr = 1.0 - c * w.sum()

        def solve(b):
            z = lu.solve(b)
            return z + w * (c * z.sum() / denom_corr)
    else:

        def solve(b):
            return lu.solve(b)

    return solve, n, adj0, deg
