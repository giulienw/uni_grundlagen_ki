from __future__ import annotations

import random
from typing import List, Tuple

from ..ga.problem import GAProblem


def pmx(parent_a: List[int], parent_b: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    size = len(parent_a)
    start, end = sorted(rng.sample(range(size), 2))
    child_a = [-1] * size
    child_b = [-1] * size

    child_a[start:end] = parent_a[start:end]
    child_b[start:end] = parent_b[start:end]

    mapping_a = {parent_b[i]: parent_a[i] for i in range(start, end)}
    mapping_b = {parent_a[i]: parent_b[i] for i in range(start, end)}

    def fill(child, donor, mapping):
        for idx in range(size):
            if start <= idx < end:
                continue
            gene = donor[idx]
            while gene in mapping:
                gene = mapping[gene]
            child[idx] = gene
        return child

    child_a = fill(child_a, parent_b, mapping_a)
    child_b = fill(child_b, parent_a, mapping_b)
    return child_a, child_b


def order_crossover(parent_a: List[int], parent_b: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    size = len(parent_a)
    start, end = sorted(rng.sample(range(size), 2))
    child_a = [-1] * size
    child_b = [-1] * size

    child_a[start:end] = parent_a[start:end]
    child_b[start:end] = parent_b[start:end]

    def fill(child, parent):
        pos = end % size
        parent_idx = end % size
        while -1 in child:
            gene = parent[parent_idx % size]
            if gene not in child:
                child[pos % size] = gene
                pos = (pos + 1) % size
            parent_idx = (parent_idx + 1) % size
        return child

    child_a = fill(child_a, parent_b)
    child_b = fill(child_b, parent_a)
    return child_a, child_b


class EightQueensProblem(GAProblem):
    target_fitness = 28.0

    def __init__(self, crossover_operator: str = "pmx", mutation_operator: str = "swap"):
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.size = 8
        self.total_pairs = self.size * (self.size - 1) // 2

    def random_individual(self, rng: random.Random) -> List[int]:
        perm = list(range(self.size))
        rng.shuffle(perm)
        return perm

    def fitness(self, individual: List[int]) -> float:
        conflicts = 0
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if abs(individual[i] - individual[j]) == abs(i - j):
                    conflicts += 1
        return self.total_pairs - conflicts

    def crossover(self, parent_a: List[int], parent_b: List[int], rng: random.Random) -> tuple[List[int], List[int]]:
        if self.crossover_operator == "ox":
            return order_crossover(parent_a, parent_b, rng)
        return pmx(parent_a, parent_b, rng)

    def mutate(self, individual: List[int], rng: random.Random, mutation_rate: float) -> List[int]:
        child = individual[:]
        if self.mutation_operator == "insert":
            if rng.random() < mutation_rate:
                i, j = sorted(rng.sample(range(self.size), 2))
                gene = child.pop(j)
                child.insert(i, gene)
        else:
            if rng.random() < mutation_rate:
                i, j = rng.sample(range(self.size), 2)
                child[i], child[j] = child[j], child[i]
        return child

    def decode(self, individual: List[int]):
        return individual[:]
