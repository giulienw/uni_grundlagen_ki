from __future__ import annotations

import csv
import itertools
import random
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

from .ga import GAConfig, GeneticAlgorithm, roulette_wheel_selection, tournament_selection
from .ga.engine import SelectionFn
from .ga.problem import GAProblem
from .problems import EightQueensProblem, MapColoringProblem


@dataclass
class RunRecord:
    problem: str
    selection: str
    crossover: str
    mutation_rate: float
    run_id: int
    best_fitness: float
    generations: int
    evaluations: int
    success: bool
    duration_seconds: float


@dataclass(frozen=True)
class ProblemSetup:
    selection_map: Dict[str, SelectionFn]
    crossover_variants: Sequence[str]
    mutation_rates: Sequence[float]
    base_config: GAConfig
    problem_factory: Callable[[str], GAProblem]


PROBLEM_SETUPS: Dict[str, ProblemSetup] = {
    "eight_queens": ProblemSetup(
        selection_map={"roulette": roulette_wheel_selection, "tournament": tournament_selection},
        crossover_variants=("pmx", "ox"),
        mutation_rates=(0.02, 0.1),
        base_config=GAConfig(population_size=100, crossover_rate=0.9, mutation_rate=0.05, max_generations=300, elitism=1),
        problem_factory=lambda crossover_op: EightQueensProblem(crossover_operator=crossover_op),
    ),
    "map_coloring": ProblemSetup(
        selection_map={"roulette": roulette_wheel_selection, "tournament": tournament_selection},
        crossover_variants=("uniform", "one_point"),
        mutation_rates=(0.02, 0.1),
        base_config=GAConfig(population_size=120, crossover_rate=0.9, mutation_rate=0.05, max_generations=300, elitism=2),
        problem_factory=lambda crossover_op: MapColoringProblem(crossover_operator=crossover_op),
    ),
}


def run_experiments(
    problem_name: str,
    *,
    runs: int = 100,
    base_seed: int = 42,
    append: bool = False,
    results_dir: Path | str | None = None,
) -> None:
    setup = _get_setup(problem_name)
    results_path = _ensure_results_dir(results_dir)
    runs_csv = results_path / f"{problem_name}_runs.csv"

    existing_records = load_run_records(runs_csv) if append and runs_csv.exists() else []
    existing_counts = _max_run_index(existing_records)

    new_records: List[RunRecord] = []
    combinations = _build_combinations(setup)

    for cfg_idx, (selection_name, selection_fn, crossover_name, mutation_rate) in enumerate(combinations, start=1):
        key = (selection_name, crossover_name, mutation_rate)
        start_run = existing_counts.get(key, -1) + 1
        if start_run >= runs:
            continue

        config = replace(setup.base_config, mutation_rate=mutation_rate)

        for run_id in range(start_run, runs):
            rng = random.Random(base_seed + cfg_idx * 10_000 + run_id)
            problem = setup.problem_factory(crossover_name)
            ga = GeneticAlgorithm(problem, config, selection_fn)

            t0 = time.perf_counter()
            result = ga.run(rng)
            duration = time.perf_counter() - t0

            new_records.append(
                RunRecord(
                    problem=problem_name,
                    selection=selection_name,
                    crossover=crossover_name,
                    mutation_rate=mutation_rate,
                    run_id=run_id,
                    best_fitness=result.best_fitness,
                    generations=result.generations,
                    evaluations=result.evaluations,
                    success=result.success,
                    duration_seconds=duration,
                )
            )

    combined = existing_records + new_records
    if not combined:
        print("Keine neuen Durchläufe erstellt.")
        return

    save_run_records(runs_csv, combined)
    save_summary(results_path / f"{problem_name}_summary.csv", combined)


def _build_combinations(setup: ProblemSetup):
    for selection_name, selection_fn in setup.selection_map.items():
        for crossover_name, mutation_rate in itertools.product(setup.crossover_variants, setup.mutation_rates):
            yield selection_name, selection_fn, crossover_name, mutation_rate


def _get_setup(problem_name: str) -> ProblemSetup:
    try:
        return PROBLEM_SETUPS[problem_name]
    except KeyError as exc:
        raise ValueError(f"Unbekanntes Problem: {problem_name}") from exc


def _ensure_results_dir(path: Path | str | None) -> Path:
    resolved = Path(path) if path else Path(__file__).with_name("results")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _max_run_index(records: Iterable[RunRecord]) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    for record in records:
        key = (record.selection, record.crossover, record.mutation_rate)
        counts[key] = max(counts.get(key, -1), record.run_id)
    return counts


def save_run_records(path: Path, records: List[RunRecord]) -> None:
    with path.open("w", newline="") as fh:
        fieldnames = list(asdict(records[0]).keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def save_summary(path: Path, records: List[RunRecord]) -> None:
    summary_rows = aggregate_results(records)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def aggregate_results(records: List[RunRecord]) -> List[Dict[str, float]]:
    grouped: Dict[tuple, List[RunRecord]] = {}
    for record in records:
        key = (record.problem, record.selection, record.crossover, record.mutation_rate)
        grouped.setdefault(key, []).append(record)

    summary = []
    for (problem, selection, crossover, mutation_rate), runs in grouped.items():
        best_values = [r.best_fitness for r in runs]
        generations = [r.generations for r in runs]
        evaluations = [r.evaluations for r in runs]
        durations = [r.duration_seconds for r in runs]
        successes = sum(1 for r in runs if r.success)

        summary.append(
            {
                "problem": problem,
                "selection": selection,
                "crossover": crossover,
                "mutation_rate": mutation_rate,
                "runs": len(runs),
                "success_rate": successes / len(runs),
                "mean_best_fitness": statistics.mean(best_values),
                "std_best_fitness": statistics.pstdev(best_values),
                "mean_generations": statistics.mean(generations),
                "median_generations": statistics.median(generations),
                "mean_evaluations": statistics.mean(evaluations),
                "mean_duration_s": statistics.mean(durations),
            }
        )

    return summary


def load_run_records(path: Path) -> List[RunRecord]:
    records: List[RunRecord] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(
                RunRecord(
                    problem=row["problem"],
                    selection=row["selection"],
                    crossover=row["crossover"],
                    mutation_rate=float(row["mutation_rate"]),
                    run_id=int(row["run_id"]),
                    best_fitness=float(row["best_fitness"]),
                    generations=int(row["generations"]),
                    evaluations=int(row["evaluations"]),
                    success=row["success"].lower() == "true",
                    duration_seconds=float(row["duration_seconds"]),
                )
            )
    return records


def parse_args(argv: Sequence[str]) -> tuple[List[str], int, int, bool]:
    import argparse

    parser = argparse.ArgumentParser(description="Führe GA-Experimente aus.")
    parser.add_argument("problem", nargs="*", default=["eight_queens", "map_coloring"], help="Probleme: eight_queens, map_coloring")
    parser.add_argument("--runs", type=int, default=100, help="Anzahl der Läufe pro Konfiguration")
    parser.add_argument("--seed", type=int, default=42, help="Basis-Seed für RNG")
    parser.add_argument("--append", action="store_true", help="An vorhandene Ergebnisse anhängen, anstatt neu zu starten.")
    args = parser.parse_args(list(argv))
    return args.problem, args.runs, args.seed, args.append


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    arg_list = list(argv) if argv is not None else sys.argv[1:]
    problems, runs, seed, append = parse_args(arg_list)
    for problem in problems:
        print(f"Starte Experimente für {problem} (runs={runs}) ...")
        run_experiments(problem, runs=runs, base_seed=seed, append=append)
        print(f"Experimente für {problem} abgeschlossen.")
