# Optimal Link Recommendations under Local Budgets for PageRank Fairness

This repository contains the implementation of fairness-aware link recommendations for PageRank, as described in our paper [**Optimal Link Recommendations under Local Budgets for PageRank Fairness**](https://openreview.net/forum?id=EpvmjulWCl).

## Installation

### Requirements

- Python 3.10 or newer
- `uv` for environment and dependency management
- Git

Clone the repository and install all required dependencies:

```sh
git clone https://github.com/UoC-tail/fair-pagerank.git
cd fair-pagerank
uv sync
```

`uv sync` creates the project environment and installs the dependencies specified by the project.

All linear programs use the open-source HiGHS solver by default. Optionally, Gurobi can be used as an alternative solver (`uv sync --extra gurobi`, then `--solver GUROBI` for usage). A suitable Gurobi installation and license are required. 

## Data and Results

- **Datasets** are downloaded to `./data/` when first needed. The available datasets are `blogs`, `tolokers`, `pokec`, `twitter`, `github_female`, `github_male`, and `nba`. Tolokers and Pokec are rebuilt from their original public sources.
- **Results** are saved for each `<dataset>` to `./results/<dataset>/<experiment>.csv`.

Both directories are created automatically and can be changed with `--data-dir <path>` and `--results-dir <path>`.

## Usage

Each experiment can be run through the command-line interface:

```sh
uv run fair-pagerank <experiment> --dataset <dataset>
```

Datasets: `blogs`, `tolokers`, `pokec`, `twitter`, `github_female`, `github_male`, `nba`.

### Available Experiments

- `local`: target-group mass after adding up to `b` edges per source (comparing static greedy, greedy FREC+, policy iteration)
- `global`: target-group mass after adding exactly `B` edges in total (comparing gain ranking, relaxed QCLP, random subset)
- `controls`: trivial and generic recommenders
- `seeds`: variability across source samples
- `ablation`: contribution of the dangling-source gain and of policy iteration
- `runtime`: wall time of greedy FREC+ and policy iteration
- `parity`: stopping at parity under a global budget
- `exact`: exact optima at small global budgets
- `stats`: dataset and policy-iteration statistics
- `conventions`: alternative dangling-node conventions

For a list of all available options for an experiment, run:

`uv run fair-pagerank <experiment> --help` 

Run `parity` and `stats` with `OPENBLAS_NUM_THREADS=1` to reproduce the paper's values exactly.

### Computing Through Code

The following Python code loads the `blogs` dataset, samples source nodes, generates candidate edges within four hops, and computes policy-optimal recommendations under a local budget of 10 edges per source node.

```python
import numpy as np

from fair_pagerank.candidates import candidate_edges, sample_source_nodes
from fair_pagerank.datasets.registry import load_dataset
from fair_pagerank.pagerank import group_mass, mass_after_flip, pagerank
from fair_pagerank.policy import maximize_group_mass

ds = load_dataset("blogs")
adj, mask, tel = ds["adj"], ds["mask"], ds["teleport"]

sources = sample_source_nodes(adj, ratio=0.1, seed=0)
candidates = candidate_edges(adj, sources, max_dist=4)
budget = np.full(ds["n"], 10)  # at most 10 new out-links per node

edges, _, _ = maximize_group_mass(adj, 0.85, candidates, budget, tel, mask)
print(f"F before: {group_mass(pagerank(adj, 0.85, tel), mask):.3f}")  # 0.334
print(f"F after:  {mass_after_flip(adj, edges, mask, 0.85, tel):.3f}")  # 0.477
```

For the example above, the initial and final target-group PageRank masses are approximately:

F before: 0.334  
F after: 0.477

## Repository Structure and Extensibility

The main source code is located in `src/fair_pagerank/`:

```
src/fair_pagerank/
├── cli.py            command-line entry point
├── pagerank.py       PageRank with teleporting dangling nodes
├── candidates.py     candidate edges (four-hop neighbourhoods of sampled sources)
├── gain.py           exact single-edge gains
├── selection.py      static greedy and greedy FREC+
├── policy.py         policy iteration under local budgets
├── qclp.py           relaxed QCLP for a global budget
├── recommenders.py   trivial and generic control recommenders
├── absorbing.py      absorbing dangling convention (robustness check)
├── graph.py          graph utilities
├── datasets/         registry, loaders, downloads, Tolokers/Pokec preprocessing
└── experiments/      one module per experiment
```

- To add a **custom dataset**, register a loader in `datasets/registry.py`. The loader should provide the graph, target-group mask, teleport vector, and the other information expected by the dataset interface.
- To add a **custom experiment**, create a module in `src/fair_pagerank/experiments/` with `add_arguments(parser)` and `run(args)` and register it in `src/fair_pagerank/experiments/__init__.py`.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{TODO,
  title = {Optimal Link Recommendations under Local Budgets for PageRank Fairness},
  author = {{\"U}n{\"u}var, Asli Tugce and Antonelli, Simone and Bojchevski, Aleksandar},
  booktitle = {TODO},
  year = {2026},
  url = {https://openreview.net/forum?id=EpvmjulWCl},
}
```

## License

This project is licensed under the MIT License.
