from __future__ import annotations

import random
from typing import Dict, List

from ..ga.problem import GAProblem

REGIONS = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
ADJACENCY = {
    "WA": ["NT", "SA"],
    "NT": ["WA", "SA", "Q"],
    "SA": ["WA", "NT", "Q", "NSW", "V"],
    "Q": ["NT", "SA", "NSW"],
    "NSW": ["SA", "Q", "V"],
    "V": ["SA", "NSW", "T"],
    "T": ["V"],
}


class MapColoringProblem(GAProblem):
    def __init__(
        self,
        initial_colors: int = 5,
        min_colors: int = 4,
        crossover_operator: str = "uniform",
        conflict_penalty: float = 10.0,
        color_penalty: float = 1.0,
    ):
        self.initial_colors = initial_colors
        self.min_colors = min_colors
        self.crossover_operator = crossover_operator
        self.conflict_penalty = conflict_penalty
        self.color_penalty = color_penalty
        self.regions = REGIONS
        self.region_index = {name: idx for idx, name in enumerate(self.regions)}
        self.edges = self._build_edges()
        self.max_score = conflict_penalty * len(self.edges) + color_penalty * initial_colors
        self.target_fitness = self.max_score

    def _build_edges(self):
        edges = set()
        for region, neighbors in ADJACENCY.items():
            for neighbor in neighbors:
                edge = tuple(sorted((region, neighbor)))
                edges.add(edge)
        return sorted(edges)

    def random_individual(self, rng: random.Random) -> List[int]:
        return [rng.randrange(self.initial_colors) for _ in self.regions]

    def fitness(self, individual: List[int]) -> float:
        conflicts = 0
        for idx_a, idx_b in [self._indices(edge) for edge in self.edges]:
            if individual[idx_a] == individual[idx_b]:
                conflicts += 1

        colors_used = len(set(individual))
        overuse = max(0, colors_used - self.min_colors)
        penalty = conflicts * self.conflict_penalty + overuse * self.color_penalty
        return self.max_score - penalty

    def _indices(self, edge):
        a, b = edge
        return self.region_index[a], self.region_index[b]

    def crossover(self, parent_a: List[int], parent_b: List[int], rng: random.Random):
        size = len(parent_a)
        if self.crossover_operator == "one_point":
            point = rng.randrange(1, size)
            child_a = parent_a[:point] + parent_b[point:]
            child_b = parent_b[:point] + parent_a[point:]
        else:
            child_a = []
            child_b = []
            for gene_a, gene_b in zip(parent_a, parent_b):
                if rng.random() < 0.5:
                    child_a.append(gene_a)
                    child_b.append(gene_b)
                else:
                    child_a.append(gene_b)
                    child_b.append(gene_a)
        return child_a, child_b

    def mutate(self, individual: List[int], rng: random.Random, mutation_rate: float):
        child = individual[:]
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                current = child[idx]
                choices = [c for c in range(self.initial_colors) if c != current]
                if choices:
                    child[idx] = rng.choice(choices)
        return child

    def decode(self, individual: List[int]) -> Dict[str, int]:
        return dict(zip(self.regions, individual))
