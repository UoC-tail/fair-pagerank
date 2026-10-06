"""Dataset registry with the paper's defaults."""

from dataclasses import dataclass
from functools import partial

import numpy as np

from fair_pagerank.datasets.loaders import load_blogs, load_fairprrec, load_nba, load_npz


@dataclass(frozen=True)
class DatasetSpec:
    loader: object
    # Group whose PageRank mass is maximized: the group under-represented
    # relative to its size. "minority" = the smaller sensitive group;
    # "majority" = its complement (Twitter and Pokec, where the minority is
    # already over-represented).
    target: str
    display: str


DATASETS = {
    "blogs": DatasetSpec(load_blogs, "minority", "Blogs"),
    "tolokers": DatasetSpec(partial(load_npz, "tolokers"), "minority", "Tolokers"),
    "pokec": DatasetSpec(partial(load_npz, "pokec"), "majority", "Pokec"),
    "twitter": DatasetSpec(partial(load_fairprrec, "twitter"), "majority", "Twitter"),
    "github_female": DatasetSpec(partial(load_fairprrec, "github_female"), "minority", "GitHub-F"),
    "github_male": DatasetSpec(partial(load_fairprrec, "github_male"), "minority", "GitHub-M"),
    "nba": DatasetSpec(load_nba, "minority", "NBA"),
}
TARGETS = ("minority", "majority")


def load_dataset(name, data_dir="data", target=None):
    """Load a registered dataset.

    Returns a dict with
      adj       CSR float64 adjacency (u -> v), dangling nodes kept
      n         number of nodes
      sens      sensitive attribute (binary)
      minority  boolean mask of the smaller sensitive group
      mask      boolean mask of the target group (registry default unless
                `target` overrides it)
      target    "minority" or "majority"
      teleport  uniform teleport vector
    """
    spec = DATASETS[name]
    adj, sens = spec.loader(data_dir=data_dir)
    adj = adj.tocsr().astype(np.float64)
    ids, counts = np.unique(sens, return_counts=True)
    if len(ids) != 2:
        raise ValueError(f"{name}: expected a binary sensitive attribute, got {len(ids)} values")
    minority = sens == ids[np.argmin(counts)]

    target = target or spec.target
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; choose from {TARGETS}")
    mask = minority if target == "minority" else ~minority
    n = adj.shape[0]
    return {
        "adj": adj,
        "n": n,
        "sens": sens,
        "minority": minority,
        "mask": mask,
        "target": target,
        "teleport": np.ones(n) / n,
    }
