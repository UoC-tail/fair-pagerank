"""Control recommenders: trivial random-into-target edges and generic link
predictors (preferential attachment, Jaccard, Adamic-Adar)."""

import numpy as np


def trivial_edges(cand, mask, budget, seed):
    """Each source adds up to `budget` candidate edges to RANDOM target-group
    nodes (no optimization)."""
    into_target = cand[mask[cand[:, 1]]]
    order = np.argsort(into_target[:, 0], kind="stable")
    bounds = np.searchsorted(into_target[order, 0], np.unique(into_target[:, 0]))
    rng = np.random.RandomState(seed)
    picks = []
    for a, b in zip(bounds, np.append(bounds[1:], len(order)), strict=True):
        grp = order[a:b]
        picks.append(grp[rng.choice(len(grp), min(budget, len(grp)), replace=False)])
    return into_target[np.concatenate(picks)]


def score_edges(adj, cand):
    """pa / jaccard / adamic_adar scores of candidate edges (undirected
    neighborhoods for Jaccard and Adamic-Adar, target in-degree for pa)."""
    und = (adj + adj.T).astype(bool).tocsr().astype(np.float64)
    d_und = np.asarray(und.sum(1)).ravel()
    d_in = np.asarray(adj.sum(0)).ravel()
    n = adj.shape[0]
    pa, jc, ada = (np.zeros(len(cand)) for _ in range(3))
    order = np.argsort(cand[:, 0], kind="stable")
    i = 0
    while i < len(order):
        u = cand[order[i], 0]
        j = i
        while j < len(order) and cand[order[j], 0] == u:
            j += 1
        idx = order[i:j]
        tgts = cand[idx, 1]
        nbrs = und.indices[und.indptr[u] : und.indptr[u + 1]]
        vec = np.zeros(n)
        vec[nbrs] = 1.0
        cn = und @ vec
        w = np.zeros(n)
        with np.errstate(divide="ignore"):
            w[nbrs] = 1.0 / np.log(np.maximum(d_und[nbrs], 2.0))
        aa = und @ w
        pa[idx] = d_in[tgts]
        union = d_und[u] + d_und[tgts] - cn[tgts]
        jc[idx] = np.where(union > 0, cn[tgts] / np.maximum(union, 1), 0.0)
        ada[idx] = aa[tgts]
        i = j
    return {"pa": pa, "jaccard": jc, "adamic_adar": ada}


def top_k_per_source(cand, scores, k):
    """The k highest-scoring candidates of every source."""
    rows = []
    for u in np.unique(cand[:, 0]):
        m = np.where(cand[:, 0] == u)[0]
        rows.append(cand[m[np.argsort(-scores[m])[:k]]])
    return np.vstack(rows)
