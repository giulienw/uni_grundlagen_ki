from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

from .problem import GAProblem

SelectionFn = Callable[[Sequence[Any], Sequence[float], random.Random], Any]


@dataclass
class GAConfig:
    population_size: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    max_generations: int = 200
    elitism: int = 1
    stop_at_optimum: bool = True


@dataclass
class GAResult:
    best_individual: Any
    best_fitness: float
    generations: int
    evaluations: int
    fitness_history: List[float]
    success: bool
    best_generation: int


class GeneticAlgorithm:
    def __init__(self, problem: GAProblem, config: GAConfig, selection_fn: SelectionFn):
        self.problem = problem
        self.config = config
        self.selection_fn = selection_fn

    def run(self, rng: random.Random | None = None) -> GAResult:
        rng = rng or random.Random()
        population = [self.problem.random_individual(rng) for _ in range(self.config.population_size)]
        fitness_values = [self.problem.fitness(ind) for ind in population]
        evaluations = len(population)

        best_idx = max(range(len(population)), key=lambda idx: fitness_values[idx])
        best_overall = self.problem.clone(population[best_idx])
        best_fitness = fitness_values[best_idx]
        fitness_history = [best_fitness]
        best_generation = 0

        if self.config.stop_at_optimum and self.problem.is_optimal(best_fitness):
            return GAResult(best_overall, best_fitness, 0, evaluations, fitness_history, True, 0)

        for generation in range(1, self.config.max_generations + 1):
            new_population: List[Any] = []

            if self.config.elitism > 0:
                elite_pairs = sorted(zip(population, fitness_values), key=lambda pair: pair[1], reverse=True)
                for elite_individual, _ in elite_pairs[: self.config.elitism]:
                    new_population.append(self.problem.clone(elite_individual))

            while len(new_population) < self.config.population_size:
                parent_a = self.selection_fn(population, fitness_values, rng)
                parent_b = self.selection_fn(population, fitness_values, rng)

                if rng.random() < self.config.crossover_rate:
                    child_a, child_b = self.problem.crossover(parent_a, parent_b, rng)
                else:
                    child_a = self.problem.clone(parent_a)
                    child_b = self.problem.clone(parent_b)

                child_a = self.problem.mutate(child_a, rng, self.config.mutation_rate)
                child_b = self.problem.mutate(child_b, rng, self.config.mutation_rate)

                new_population.append(child_a)
                if len(new_population) < self.config.population_size:
                    new_population.append(child_b)

            population = new_population[: self.config.population_size]
            fitness_values = [self.problem.fitness(ind) for ind in population]
            evaluations += len(population)

            gen_best_idx = max(range(len(population)), key=lambda idx: fitness_values[idx])
            gen_best_fit = fitness_values[gen_best_idx]
            fitness_history.append(gen_best_fit)

            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_overall = self.problem.clone(population[gen_best_idx])
                best_generation = generation

            if self.config.stop_at_optimum and self.problem.is_optimal(gen_best_fit):
                return GAResult(best_overall, best_fitness, generation, evaluations, fitness_history, True, generation)

        success = self.problem.is_optimal(best_fitness) if self.config.stop_at_optimum else False
        return GAResult(best_overall, best_fitness, self.config.max_generations, evaluations, fitness_history, success, best_generation)
