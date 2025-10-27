from __future__ import annotations

import copy
from typing import Any, Sequence


class GAProblem:
    target_fitness: float | None = None

    def random_individual(self, rng) -> Any:
        raise NotImplementedError

    def fitness(self, individual: Any) -> float:
        raise NotImplementedError

    def crossover(self, parent_a: Any, parent_b: Any, rng) -> tuple[Any, Any]:
        raise NotImplementedError

    def mutate(self, individual: Any, rng, mutation_rate: float) -> Any:
        raise NotImplementedError

    def clone(self, individual: Any) -> Any:
        return copy.deepcopy(individual)

    def is_optimal(self, fitness_value: float) -> bool:
        if self.target_fitness is None:
            return False
        return fitness_value >= self.target_fitness

    def decode(self, individual: Any) -> Sequence[Any] | Any:
        return individual
