"""Rebuild the Tolokers-Fair and Pokec-Fair graphs from their public sources.

The two benchmarks were introduced by Loveland & Koutra, "Unveiling the Impact of
Local Homophily on GNN Fairness: In-Depth Analysis and New Benchmarks" (2024,
arXiv:2410.04287). This module is an independent re-implementation of the
procedure described in that paper; it downloads the raw data and writes
``<data_dir>/<name>/<name>.npz`` with the arrays

    edge_index  int64  (2, m)  directed edges, in source-file order
    sens        int64  (n,)    binary sensitive attribute
    labels      int64  (n,)    node classes
    features    float32 (n, d) node features

Tolokers-Fair (source: yandex-research/heterophilous-graphs, MIT licence)
    The graph is taken as is: 11,758 nodes, 519,000 edges (each pair stored once).
    ``sens`` is the last node-feature column (dominant language English = 1),
    ``labels`` is the original "banned" target.

Pokec-Fair (source: SNAP soc-Pokec)
    1. Users are ordered by user id (ids are exactly 1..N, so row = id - 1).
    2. Keep users whose "I_am_working_in_field" entry is one of the five classes
       named in the paper (service work, construction, finance, healthcare,
       transport; Slovak strings in ``POKEC_JOBS``) and whose gender is known.
    3. Keep friendships whose two endpoints both survived, in file order, and
       renumber the surviving users 0..n-1 in user-id order.
    4. Keep the largest weakly connected component, again preserving the order
       of nodes and edges.
    ``sens`` = gender (1 = male), ``labels`` = position of the job in the
    alphabetically sorted class list. Features (computed before step 4) are:
    public, completion/100, gender, last login and registration rescaled to
    [0, 1] between the earliest registration and the crawl date (2012-05-28),
    age/100, and a one-hot encoding of the region.

Usage:  python preprocess_loveland.py DATA_DIR [--raw-dir DIR] [--only NAME ...]
"""

from __future__ import annotations

import csv
import hashlib
import os
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

TOLOKERS_URL = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/tolokers.npz"
SNAP_URL = "https://snap.stanford.edu/data/"
POKEC_FILES = ("soc-pokec-profiles.txt.gz", "soc-pokec-relationships.txt.gz")

# Pokec-Fair classes: transport, finance, service work, construction, healthcare.
# Sorted alphabetically; a node's label is the index of its job in this tuple.
POKEC_JOBS = (
    "dopravy",
    "ekonomiky a financii",
    "sluzieb a obchodu",
    "stavebnictva",
    "zdravotnictva",
)
CRAWL_END = np.datetime64("2012-05-28T00:00:00", "s")
# Columns of soc-pokec-profiles we need (0-based) and the names we give them.
PROFILE_COLS = {
    0: "user_id",
    1: "public",
    2: "completion",
    3: "gender",
    4: "region",
    5: "last_login",
    6: "registration",
    7: "age",
    9: "job",
}


# SHA-256 of the raw files the reference outputs were verified against (Oct 2026).
SHA256 = {
    "tolokers.npz": "dacf3ac94cec53d03cd2adb5255c08b33dee1656c33ca8164a464bd9450a1667",
    "soc-pokec-profiles.txt.gz": "d4c6b0151cfa3571a7864ea12b06a7c5367e33ecbfe20358cf5420d76e40bad9",
    "soc-pokec-relationships.txt.gz": (
        "1a23e0ec8a4e497752125f6b3f01696fea7fcdb696fa61d1e822faf4d0d69b14"
    ),
}


def download(url: str, dest: Path) -> Path:
    """Fetch ``url`` to ``dest`` unless the file already exists, then warn if
    its checksum differs from the one the outputs were verified with."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    digest = hashlib.sha256()
    with open(dest, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    if SHA256.get(dest.name, digest.hexdigest()) != digest.hexdigest():
        warnings.warn(
            f"{dest} differs from the verified upstream file; "
            "outputs may not match the published benchmark",
            stacklevel=2,
        )
    return dest


def _save(out_path: Path, **arrays) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    n, m = len(arrays["sens"]), arrays["edge_index"].shape[1]
    print(f"wrote {out_path}  (n={n}, m={m})", flush=True)
    return out_path


def _induced_subgraph(edge_index: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Edges with both ends in the boolean mask ``keep``, nodes renumbered
    by their rank among kept nodes; the original edge order is preserved."""
    new_id = np.full(keep.shape[0], -1, dtype=np.int64)
    new_id[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    inside = keep[edge_index[0]] & keep[edge_index[1]]
    return new_id[edge_index[:, inside]]


def largest_weak_component(edge_index: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask of the nodes in the largest weakly connected component."""
    adj = sparse.coo_matrix(
        (np.ones(edge_index.shape[1], dtype=np.int8), (edge_index[0], edge_index[1])), shape=(n, n)
    ).tocsr()
    _, comp = connected_components(adj, directed=True, connection="weak")
    return comp == np.bincount(comp).argmax()


# --------------------------------------------------------------------------- #
# Tolokers-Fair
# --------------------------------------------------------------------------- #
def build_tolokers(data_dir, raw_dir=None) -> Path:
    """Write ``<data_dir>/tolokers/tolokers.npz`` from the Yandex release."""
    data_dir = Path(data_dir)
    raw_dir = Path(raw_dir) if raw_dir else data_dir / "tolokers" / "raw"
    raw = np.load(download(TOLOKERS_URL, raw_dir / "tolokers.npz"))
    features = raw["node_features"].astype(np.float32)
    return _save(
        data_dir / "tolokers" / "tolokers.npz",
        edge_index=np.ascontiguousarray(raw["edges"].T, dtype=np.int64),
        sens=features[:, -1].astype(np.int64),
        labels=raw["node_labels"].astype(np.int64),
        features=features,
    )


# --------------------------------------------------------------------------- #
# Pokec-Fair
# --------------------------------------------------------------------------- #
def _read_profiles(path: Path) -> pd.DataFrame:
    """Selected profile columns as raw strings ('null' marks a missing value),
    sorted by user id."""
    prof = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=list(PROFILE_COLS),
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )
    prof = prof.rename(columns=PROFILE_COLS)
    prof["user_id"] = prof["user_id"].astype(np.int64)
    prof = prof.sort_values("user_id", kind="stable").reset_index(drop=True)
    if not np.array_equal(prof["user_id"].to_numpy(), np.arange(1, len(prof) + 1)):
        raise ValueError("expected Pokec user ids to be exactly 1..N")
    return prof


def _read_relationships(path: Path) -> np.ndarray:
    """Directed friendships as a (2, m) array of 0-based user positions."""
    rel = pd.read_csv(path, sep="\t", header=None, dtype=np.int64).to_numpy()
    return np.ascontiguousarray(rel.T) - 1


def _to_seconds(stamps: np.ndarray) -> np.ndarray:
    """'YYYY-MM-DD hh:mm:ss.f' strings -> integer seconds since the epoch."""
    ms = stamps.astype("datetime64[ms]").astype(np.int64)
    return ms // 1000


def _pokec_features(prof: pd.DataFrame) -> np.ndarray:
    """Numeric and one-hot region features for the (already filtered) users."""
    login = _to_seconds(prof["last_login"].to_numpy())
    reg = _to_seconds(prof["registration"].to_numpy())
    start, end = reg.min(), CRAWL_END.astype(np.int64)
    numeric = np.column_stack(
        [
            prof["public"].to_numpy().astype(np.float64),
            prof["completion"].astype(np.int64).to_numpy() / 100,
            prof["gender"].to_numpy().astype(np.float64),
            (login - start) / (end - start),
            (reg - start) / (end - start),
            prof["age"].astype(np.int64).to_numpy() / 100,
        ]
    )
    regions, region_idx = np.unique(prof["region"].to_numpy().astype(str), return_inverse=True)
    one_hot = np.zeros((len(prof), len(regions)), dtype=np.float64)
    one_hot[np.arange(len(prof)), region_idx] = 1.0
    return np.hstack([numeric, one_hot]).astype(np.float32)


def build_pokec(data_dir, raw_dir=None) -> Path:
    """Write ``<data_dir>/pokec/pokec.npz`` from the SNAP soc-Pokec dump."""
    data_dir = Path(data_dir)
    raw_dir = Path(raw_dir) if raw_dir else data_dir / "pokec" / "raw"
    prof_path, rel_path = (download(SNAP_URL + f, raw_dir / f) for f in POKEC_FILES)

    prof = _read_profiles(prof_path)
    keep = prof["job"].isin(POKEC_JOBS).to_numpy() & prof["gender"].isin(["0", "1"]).to_numpy()
    prof = prof[keep].reset_index(drop=True)
    edge_index = _induced_subgraph(_read_relationships(rel_path), keep)
    features = _pokec_features(prof)

    in_lcc = largest_weak_component(edge_index, len(prof))
    edge_index = _induced_subgraph(edge_index, in_lcc)
    prof, features = prof[in_lcc].reset_index(drop=True), features[in_lcc]

    labels = np.searchsorted(np.array(POKEC_JOBS), prof["job"].to_numpy().astype(str))
    return _save(
        data_dir / "pokec" / "pokec.npz",
        edge_index=edge_index,
        sens=(prof["gender"].to_numpy() == "1").astype(np.int64),
        labels=labels.astype(np.int64),
        features=features,
        label_names=np.array(POKEC_JOBS),
    )
