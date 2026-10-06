"""Command-line entry point: `fair-pagerank <experiment> --dataset <name> [options]`."""

import argparse
import logging
import warnings

from fair_pagerank.datasets import DATASETS, TARGETS
from fair_pagerank.experiments import EXPERIMENTS


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fair-pagerank",
        description="Reproduce the experiments of 'Optimal Link Recommendations under "
        "Local Budgets for PageRank Fairness' (LoG 2026). Each experiment writes "
        "<results-dir>/<dataset>/<experiment>.csv.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", required=True, choices=list(DATASETS))
    common.add_argument(
        "--target",
        choices=TARGETS,
        default=None,
        help="override the dataset's target group (default: the paper's choice; "
        "results then go to <results-dir>/<dataset>_<target>/)",
    )
    common.add_argument("--data-dir", default="data", help="dataset directory (default: data)")
    common.add_argument(
        "--results-dir", default="results", help="output directory (default: results)"
    )

    sub = parser.add_subparsers(dest="experiment", required=True, metavar="<experiment>")
    for name, module in EXPERIMENTS.items():
        p = sub.add_parser(
            name,
            parents=[common],
            help=module.__doc__.strip().splitlines()[0],
            description=module.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(p)
        p.set_defaults(run=module.run)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    # graph_cert builds its LP with the deprecated `*` matrix product of cvxpy
    warnings.filterwarnings("ignore", message=r"\s*This use of ``\*``", category=UserWarning)
    args.run(args)


if __name__ == "__main__":
    main()
