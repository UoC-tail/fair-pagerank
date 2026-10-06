"""Gain-based edge selection: static top-k and the round-based greedy FREC+."""

import logging

import numpy as np
import scipy.sparse as sp
from graph_cert.utils import flip_edges, top_k_numba

from fair_pagerank.gain import edge_gains
from fair_pagerank.pagerank import pagerank

log = logging.getLogger(__name__)


def select_top_k_local(edges, n_nodes, gains, budget_per_node):
    """Top-k positive-gain edges per source node (static greedy)."""
    nodes = np.sort(np.unique(edges[:, 0]))
    positive = gains > 0
    if not positive.any():
        return np.empty((0, 2), dtype=int)
    gain_matrix = sp.csr_matrix(
        (gains[positive], (edges[positive, 0], edges[positive, 1])),
        shape=(n_nodes, n_nodes),
    )
    if np.isscalar(budget_per_node):
        budget_per_node = np.full(n_nodes, budget_per_node, dtype=int)
    top_k = top_k_numba(gain_matrix[nodes], budget_per_node[nodes])
    return np.column_stack((nodes[top_k[:, 0]], top_k[:, 1]))


def select_top_k_global(edges, gains, budget):
    """Top-`budget` positive-gain edges overall (rank once by gain)."""
    positive = gains > 0
    if not positive.any():
        return np.empty((0, 2), dtype=int)
    pos_edges, pos_gains = edges[positive], gains[positive]
    k = min(budget, len(pos_edges))
    return pos_edges[np.argsort(pos_gains)[-k:][::-1]]


def greedy_frec(adj, alpha, mask, candidates, rounds=10):
    """Round-based greedy FREC (Tsioutsiouliklis et al., WWW'22): in every
    round each source adds its top-1 positive-gain candidate, then PageRank
    and gains are recomputed. With `edge_gains` scoring dangling sources this
    is FREC+.

    Returns (all added edges, F after 0..rounds rounds).
    """
    n = adj.shape[0]
    teleport = np.ones(n) / n
    adj = adj.tocsr().astype(np.float64)
    selected = []
    sources = np.unique(candidates[:, 0])
    pr_history = [pagerank(adj, alpha, teleport)[mask].sum()]

    for r in range(rounds):
        if len(candidates) == 0:
            pr_history.append(pr_history[-1])
            continue
        pi = pagerank(adj, alpha, teleport)
        gains = edge_gains(candidates, adj, pi, mask, alpha)

        round_edges = []
        for src in sources:
            sel = candidates[:, 0] == src
            if not np.any(sel):
                continue
            best = gains[sel].argmax()
            if gains[sel][best] > 0:
                round_edges.append(candidates[sel][best])

        if not round_edges:
            pr_history.append(pr_history[-1])
            continue

        round_edges = np.array(round_edges)
        selected.append(round_edges)
        adj = flip_edges(adj, round_edges).tocsr()
        keys = candidates[:, 0].astype(np.int64) * n + candidates[:, 1]
        rm = round_edges[:, 0].astype(np.int64) * n + round_edges[:, 1]
        candidates = candidates[~np.isin(keys, rm)]
        pr_history.append(pagerank(adj, alpha, teleport)[mask].sum())
        log.info(
            "FREC+ round %d/%d: %d edges, F=%.6f", r + 1, rounds, len(round_edges), pr_history[-1]
        )

    all_edges = np.vstack(selected) if selected else np.empty((0, 2), dtype=int)
    return all_edges, np.array(pr_history)
