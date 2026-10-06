"""Rebuild Tolokers and Pokec from their original public sources.

Output contract (consumed by `loaders.load_npz`): data/<name>/<name>.npz with
  edge_index : int array, shape (2, m), directed edges in stored order
  sens       : array, shape (n,), binary sensitive attribute per node

Sources and benchmark definitions:
  Tolokers: Yandex heterophilous-graphs (Platonov et al., ICLR 2023, MIT),
            Tolokers-Fair definition of Loveland & Koutra (2024).
  Pokec:    SNAP soc-Pokec (Takac & Zabovsky, 2012),
            Pokec-Fair definition of Loveland & Koutra (2024).
"""

from fair_pagerank.datasets.loveland import build_pokec, build_tolokers

# Called by the loader whenever data/<name>/<name>.npz is missing.
PREPROCESSORS = {"tolokers": build_tolokers, "pokec": build_pokec}


def build(name, data_dir):
    """Create data_dir/<name>/<name>.npz from the original public sources."""
    if name not in PREPROCESSORS:
        raise FileNotFoundError(
            f"{data_dir}/{name}/{name}.npz not found and no preprocessing is "
            f"registered for '{name}' yet (see fair_pagerank/datasets/preprocess.py). "
            f"Place the preprocessed file there manually."
        )
    PREPROCESSORS[name](data_dir)
