from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .experiments import run_experiments


def run_problem_cli(
    argv: Sequence[str] | None,
    *,
    problem_name: str,
    description: str,
    results_dir: Path | None = None,
) -> None:
    args = _parse_args(argv, description)
    target_dir = results_dir or (Path.cwd() / "results")
    target_dir.mkdir(parents=True, exist_ok=True)

    run_experiments(
        problem_name,
        runs=args.runs,
        base_seed=args.seed,
        append=args.append,
        results_dir=target_dir,
    )


def _parse_args(argv: Sequence[str] | None, description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--runs", type=int, default=100, help="Anzahl der Läufe pro Konfiguration")
    parser.add_argument("--seed", type=int, default=42, help="Basis-Seed für RNG")
    parser.add_argument("--append", action="store_true", help="An vorhandene Ergebnisse anhängen.")

    args_list = list(argv) if argv is not None else sys.argv[1:]
    return parser.parse_args(args_list)
