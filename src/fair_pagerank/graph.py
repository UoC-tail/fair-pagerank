"""Small structural graph utilities."""

import logging

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)


def out_degree(adj):
    return np.asarray(adj.tocsr().sum(axis=1)).ravel()


def minimal_dangling_repair(adj):
    """Give every zero-out-degree node one out-edge: the reverse of the in-edge
    from its highest-out-degree in-neighbor (non-dangling by construction).
    Adds exactly #dangling edges and leaves all other rows untouched.

    Only used for graph_cert's relaxed QCLP / upper bounds, which divide by
    the out-degree and cannot handle dangling rows.
    """
    adj = adj.tocsr().astype(np.int32).copy()
    n = adj.shape[0]
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    dangling = np.where(out_deg == 0)[0]
    if len(dangling) == 0:
        return adj

    adj_in = adj.T.tocsr()  # adj_in[u].indices = in-neighbors of u
    rows, cols = [], []
    for u in dangling:
        ins = adj_in.indices[adj_in.indptr[u] : adj_in.indptr[u + 1]]
        # isolated node -> self-loop; else reverse the in-edge of the
        # highest-out-degree in-neighbor
        v = u if len(ins) == 0 else ins[np.argmax(out_deg[ins])]
        rows.append(u)
        cols.append(v)

    repair = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, n))
    adj = adj + repair
    adj.data[:] = 1
    log.info("minimal_dangling_repair: %d dangling nodes repaired", len(dangling))
    return adj


def add_self_loops(adj):
    """A + I: a self-loop on every node (existing self-loops are not doubled)."""
    adj = adj.copy()
    adj.setdiag(1)
    return adj
