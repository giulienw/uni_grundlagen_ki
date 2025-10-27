from __future__ import annotations

import random
from typing import List, Sequence


def _prepare_probs(fitness_values: Sequence[float]) -> List[float]:
    min_fit = min(fitness_values)
    shift = abs(min_fit) + 1e-9 if min_fit < 0 else 0.0
    adjusted = [(f + shift) for f in fitness_values]
    total = sum(adjusted)
    if total <= 0:
        return [1.0 / len(adjusted) for _ in adjusted]
    return [f / total for f in adjusted]


def roulette_wheel_selection(population: Sequence, fitness_values: Sequence[float], rng: random.Random):
    probs = _prepare_probs(fitness_values)
    r = rng.random()
    cumulative = 0.0
    for individual, prob in zip(population, probs):
        cumulative += prob
        if r <= cumulative:
            return individual
    return population[-1]


def tournament_selection(population: Sequence, fitness_values: Sequence[float], rng: random.Random, k: int = 3):
    indices = list(range(len(population)))
    chosen = rng.sample(indices, k)
    best_idx = max(chosen, key=lambda idx: fitness_values[idx])
    return population[best_idx]
