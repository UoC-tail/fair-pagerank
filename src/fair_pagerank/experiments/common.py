"""Shared protocol constants, setup and CSV output for all experiments."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fair_pagerank.candidates import candidate_edges, sample_source_nodes
from fair_pagerank.datasets import DATASETS, load_dataset

log = logging.getLogger(__name__)

ALPHA = 0.85  # damping factor; teleport is uniform
SOURCE_RATIO = 0.1  # fraction of nodes sampled as recommendation sources
SOURCE_SEED = 0  # seed of the source sample (all experiments but `seeds`)
MAX_HOPS = 4  # candidates: non-neighbors within 4 undirected hops
BUDGET = 10  # local budget per source


def setup(args):
    """Load the dataset selected on the command line (see cli.py)."""
    ds = load_dataset(args.dataset, args.data_dir, args.target)
    ds["name"] = args.dataset
    log.info(
        "%s: n=%d edges=%d target=%s (%d nodes, %.1f%%)",
        args.dataset,
        ds["n"],
        ds["adj"].nnz,
        ds["target"],
        ds["mask"].sum(),
        100 * ds["mask"].mean(),
    )
    return ds


def candidates(adj, seed=SOURCE_SEED):
    """4-hop candidate edges from a 10% random source sample."""
    cand = candidate_edges(adj, sample_source_nodes(adj, SOURCE_RATIO, seed), MAX_HOPS)
    log.info("candidates (seed %d): %d edges", seed, len(cand))
    return cand


def local_budget(n, budget):
    return budget * np.ones(n, dtype=int)


def write_csv(rows, args, experiment, merge_on=None):
    """Write results/<dataset>/<experiment>.csv (results dir from the CLI).

    With `merge_on`, rows already in the file whose key differs from every
    new row are kept, so independent runs (e.g. one job per seed) can share
    one file.
    """
    name = args.dataset
    if args.target not in (None, DATASETS[name].target):  # non-default target group
        name = f"{name}_{args.target}"
    out = Path(args.results_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{experiment}.csv"
    df = pd.DataFrame(rows)
    if merge_on and path.exists():
        old = pd.read_csv(path)
        new_keys = set(map(tuple, df[merge_on].astype(str).to_numpy()))
        stale = old[merge_on].astype(str).apply(tuple, axis=1).isin(new_keys)
        df = pd.concat([old[~stale], df], ignore_index=True)
    df.to_csv(path, index=False)
    log.info("saved %s\n%s", path, df.to_string(index=False))
    return path
