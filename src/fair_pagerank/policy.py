"""Policy iteration under local budgets with teleport handling of dangling nodes.

Adapted from graph_cert's policy_iteration (Bojchevski & Günnemann, NeurIPS
2019): the value solve (I - alpha T) v = reward treats zero-out-degree rows of
T as the teleport distribution, i.e. the optimizer targets exactly the
PageRank objective that is evaluated. It reduces to graph_cert's routine on
graphs without dangling nodes.
"""

import logging

import numpy as np
import scipy.sparse as sp
from graph_cert.utils import edges_to_sparse, flip_edges, top_k_numba
from scipy.sparse.linalg import gmres

log = logging.getLogger(__name__)


def _value_solve(adj_flipped, alpha, reward, teleport):
    """(I - alpha T) v = reward with dangling rows -> teleport: gmres on the
    sparse (I - alpha T0) plus a rank-1 Sherman-Morrison correction."""
    n = adj_flipped.shape[0]
    d = np.asarray(adj_flipped.sum(1)).ravel()
    dangling = d == 0
    inv_d = np.zeros(n)
    inv_d[~dangling] = 1.0 / d[~dangling]
    b = sp.eye(n) - alpha * (sp.diags(inv_d) @ adj_flipped)
    w1 = gmres(b, reward, rtol=1e-8, atol=0.0)[0]
    if not dangling.any():
        return w1
    w2 = gmres(b, dangling.astype(np.float64), rtol=1e-8, atol=0.0)[0]
    denom = 1.0 - alpha * (teleport @ w2)
    return w1 + (alpha * (teleport @ w1) / denom) * w2


def policy_iteration(adj, alpha, fragile, local_budget, reward, teleport, max_iter=50):
    """Maximize reward^T pi over subsets of `fragile` with <= local_budget[u]
    flips per source u.

    The teleport value function can make the policy cycle on dangling-heavy
    graphs instead of converging, so the best edge set seen (by its exact
    objective) is returned; iteration stops when the edge set repeats or the
    best value has not improved for 6 iterations.

    Returns (edges, objective, number of iterations).
    """
    n = adj.shape[0]
    adj = adj.tocsr()
    cur_fragile = np.array([])
    prev_fragile = np.array([[0, 0]])
    if not isinstance(local_budget, np.ndarray):
        local_budget = np.repeat(local_budget, n)
    ri = reward[fragile[:, 0]]
    # +1 if adding an edge (absent in adj), -1 if removing (present)
    add_rem = 1 - 2 * np.asarray(adj[fragile[:, 0], fragile[:, 1]]).ravel()

    best_fragile, best_obj = cur_fragile, -np.inf
    patience, stale = 6, 0
    n_iter = 0

    for _ in range(max_iter):
        n_iter += 1
        adj_flipped = flip_edges(adj, cur_fragile).tocsr()
        mean_reward = _value_solve(adj_flipped, alpha, reward, teleport)

        # objective of the CURRENT flipped graph under teleport
        obj = mean_reward @ teleport * (1 - alpha)
        if obj > best_obj + 1e-9:
            best_obj, best_fragile, stale = obj, cur_fragile, 0
        else:
            stale += 1

        vi = mean_reward[fragile[:, 0]]
        vj = mean_reward[fragile[:, 1]]
        change = (vj - ((vi - ri) / alpha)) * add_rem
        improve = change > 0
        frag = edges_to_sparse(fragile[improve], n, change[improve])
        cur_fragile = top_k_numba(frag, local_budget)

        same = (edges_to_sparse(prev_fragile, n) - edges_to_sparse(cur_fragile, n)).nnz == 0
        if same or stale >= patience:
            break
        prev_fragile = cur_fragile.copy()

    log.info(
        "policy iteration: %d iterations, %d edges, objective %.6f",
        n_iter,
        len(best_fragile),
        best_obj,
    )
    return best_fragile, best_obj, n_iter


def maximize_group_mass(adj, alpha, fragile, local_budget, teleport, mask):
    """Policy iteration with reward = 1_S (maximize the target-group mass F).

    Returns (edges, objective, number of iterations).
    """
    return policy_iteration(adj, alpha, fragile, local_budget, mask.astype(float), teleport)
