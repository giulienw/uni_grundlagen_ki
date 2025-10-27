"""Implementierung und Auswertung des Genetic Algorithmus (EA.02)."""

from .experiments import main as run_experiments
from .cli import run_problem_cli

__all__ = ["run_experiments", "run_problem_cli"]
