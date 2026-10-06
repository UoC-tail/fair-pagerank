"""One module per experiment of the paper. Each exposes add_arguments(parser)
and run(args); the module docstring documents the CSV it writes."""

from fair_pagerank.experiments import (
    ablation,
    controls,
    conventions,
    exact,
    global_budget,
    local,
    parity,
    runtime,
    seeds,
    stats,
)

EXPERIMENTS = {
    "local": local,
    "global": global_budget,
    "controls": controls,
    "seeds": seeds,
    "ablation": ablation,
    "runtime": runtime,
    "parity": parity,
    "exact": exact,
    "stats": stats,
    "conventions": conventions,
}
