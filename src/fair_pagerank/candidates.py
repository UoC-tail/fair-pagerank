"""Candidate (recommendable) edges, following the fairPRrec protocol."""

import numpy as np


def sample_source_nodes(adj, ratio=0.1, seed=0):
    """Random sample of source nodes (as fairPRrec's getSourceNodes)."""
    n = adj.shape[0]
    rng = np.random.RandomState(seed)
    k = int(ratio * n)
    return np.sort(rng.permutation(n)[:k])


def candidate_edges(adj, source_nodes, max_dist=4):
    """All non-neighbors within `max_dist` undirected hops of each source.

    Returns an [m, 2] array of (source, target) pairs, grouped by source in
    the order of `source_nodes` and with targets sorted.
    """
    adj = adj.tocsr()
    adj_undir = (adj + adj.T).astype(bool).astype(float).tocsr()

    chunks = []
    for src in source_nodes:
        existing = set(adj.indices[adj.indptr[src] : adj.indptr[src + 1]])
        existing.add(src)

        visited = {src}
        frontier = {src}
        for _ in range(max_dist):
            next_frontier = set()
            for node in frontier:
                start, end = adj_undir.indptr[node], adj_undir.indptr[node + 1]
                for nbr in adj_undir.indices[start:end]:
                    if nbr not in visited:
                        next_frontier.add(nbr)
                        visited.add(nbr)
            frontier = next_frontier

        candidates = np.array(sorted(visited - existing))
        if len(candidates) > 0:
            chunks.append(np.column_stack((np.full(len(candidates), src), candidates)))

    return np.vstack(chunks) if chunks else np.empty((0, 2), dtype=int)
