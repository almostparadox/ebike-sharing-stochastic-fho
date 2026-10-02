#!/usr/bin/env python3
"""Run full simulation replicating thesis parameters."""
from scslp.cli import cli_solve  # type: ignore

if __name__ == "__main__":
    cli_solve([
        "--datadir",
        "data",
        "--outdir",
        "results",
        "--generations",
        "100",
        "--pop-size",
        "50",
        "--seed",
        "42",
    ])
