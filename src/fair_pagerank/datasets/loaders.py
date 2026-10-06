"""Format-specific graph loaders. Each returns (adj, sens): a CSR float64
adjacency with adj[u, v] = 1 for the directed edge u -> v, and the per-node
sensitive attribute. The constructions replicate those used for the paper
exactly (including edge orientation), so results are reproduced bit-for-bit.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fair_pagerank.datasets import preprocess
from fair_pagerank.datasets.download import ensure_fairprrec_files, ensure_nba_files

log = logging.getLogger(__name__)


def _coo_adjacency(row, col, n):
    """Unit-weight adjacency from an edge list (duplicates are summed)."""
    data = np.ones(len(row), dtype=np.float32)
    return sp.coo_matrix((data, (row, col)), shape=(n, n))


def load_blogs(data_dir, download=True):
    """Political blogs as released with fairPRrec: the edge list is read as
    directed, in stored order, without LCC extraction (it is weakly
    connected and has no self-loops or duplicates)."""
    graph, community = ensure_fairprrec_files("blogs", data_dir, download)
    sens = np.loadtxt(community, skiprows=1).astype(int)[:, 1]
    edges = np.loadtxt(graph, skiprows=1).astype(int).T
    n = int(edges.max()) + 1
    return _coo_adjacency(edges[0], edges[1], n).tocsr(), sens


def load_fairprrec(name, data_dir, download=True):
    """Twitter / GitHub in the fairPRrec format: out_graph.txt (line 1 = n,
    then "src<TAB>dst") and out_community.txt (line 1 = #groups, then "node
    group"). Edge direction kept, self-loops dropped, weak LCC extracted."""
    gfile, cfile = ensure_fairprrec_files(name, data_dir, download)
    with open(gfile) as fh:
        n = int(fh.readline())
    e = np.loadtxt(gfile, skiprows=1, dtype=np.int64)
    comm = np.loadtxt(cfile, skiprows=1, dtype=np.int64)
    sens = np.zeros(n, dtype=int)
    sens[comm[:, 0]] = comm[:, 1]

    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    n_comp, cc = sp.csgraph.connected_components(A, directed=True, connection="weak")
    if n_comp > 1:
        keep = np.where(cc == np.bincount(cc).argmax())[0]
        A, sens = A[keep][:, keep], sens[keep]
        A.eliminate_zeros()
        log.info("%s: kept weak LCC %d/%d nodes (%d components)", name, len(keep), n, n_comp)
    return A.astype(np.float64), sens


def load_nba(data_dir, download=True):
    """NBA rebuilt from the raw PyGDebias relationship file, keeping the edge
    direction (the usual loaders symmetrize it). Sensitive attribute:
    nationality (`country`). Self-loops dropped, weak LCC extracted."""
    csv_path, edge_path = ensure_nba_files(data_dir, download)
    df = pd.read_csv(csv_path)
    n = len(df)
    sens = df["country"].to_numpy().astype(int)

    # raw user ids exceed 2**53: parse as int64, never as float
    e = np.genfromtxt(edge_path, dtype=np.int64)
    idx = df["user_id"].to_numpy().astype(np.int64)
    idmap = {j: i for i, j in enumerate(idx)}
    e = np.array([idmap[x] for x in e.reshape(-1)]).reshape(e.shape)

    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()

    n_comp, cc = sp.csgraph.connected_components(A, directed=False)
    if n_comp > 1:
        keep = np.where(cc == np.bincount(cc).argmax())[0]
        A = A[keep][:, keep]
        A.eliminate_zeros()
        sens = sens[keep]
        log.info("nba: kept weak LCC %d/%d nodes (%d components)", len(keep), n, n_comp)
    return A.astype(np.float64), sens


def load_npz(name, data_dir):
    """Tolokers / Pokec from data_dir/<name>/<name>.npz (`edge_index`, 2 x m,
    stored order preserved; `sens`, length n). Self-loops are dropped and the
    weak LCC is extracted, keeping the stored edge orientation and order."""
    path = Path(data_dir) / name / f"{name}.npz"
    if not path.exists():
        preprocess.build(name, data_dir)
    with np.load(path) as z:
        edge_index = np.asarray(z["edge_index"], dtype=np.int64)
        sens = np.asarray(z["sens"])

    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    n = int(edge_index.max()) + 1
    adj = _coo_adjacency(edge_index[0], edge_index[1], n)
    n_comp, labels = sp.csgraph.connected_components(adj, directed=False)
    if n_comp > 1:
        lcc = labels == np.bincount(labels).argmax()
        nodes = np.where(lcc)[0]
        node_map = -np.ones(n, dtype=int)
        node_map[nodes] = np.arange(len(nodes))
        keep = lcc[edge_index[0]] & lcc[edge_index[1]]
        edge_index = node_map[edge_index[:, keep]]
        sens = sens[nodes]
        log.info("%s: kept weak LCC %d/%d nodes (%d components)", name, len(nodes), n, n_comp)

    n = int(edge_index.max()) + 1
    return _coo_adjacency(edge_index[0], edge_index[1], n).tocsr(), sens.astype(int)
