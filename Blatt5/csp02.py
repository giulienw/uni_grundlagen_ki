from __future__ import annotations

import itertools
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

BinaryRelation = Callable[[int, int], bool]


@dataclass(frozen=True)
class BinaryConstraint:

    var1: str
    var2: str
    relation: BinaryRelation

    def involves(self, var: str) -> bool:
        return var == self.var1 or var == self.var2

    def other(self, var: str) -> str:
        if var == self.var1:
            return self.var2
        if var == self.var2:
            return self.var1
        raise ValueError(f"Variable {var} nicht im Constraint {self}")

    def is_satisfied(self, assignment: Dict[str, int]) -> bool:
        if self.var1 not in assignment or self.var2 not in assignment:
            return True
        return self.relation(assignment[self.var1], assignment[self.var2])

    def is_satisfied_with(self, var: str, value: int, assignment: Dict[str, int]) -> bool:
        if var == self.var1:
            val2 = assignment.get(self.var2)
            if val2 is None:
                return True
            return self.relation(value, val2)
        if var == self.var2:
            val1 = assignment.get(self.var1)
            if val1 is None:
                return True
            return self.relation(val1, value)
        raise ValueError(f"Variable {var} nicht im Constraint {self}")


class CSP:
    def __init__(self, variables: Sequence[str], domains: Dict[str, Iterable[int]]):
        self.variables: List[str] = list(variables)
        self.domains: Dict[str, Set[int]] = {var: set(values) for var, values in domains.items()}
        self.constraints_by_var: Dict[str, List[BinaryConstraint]] = {var: [] for var in self.variables}
        self.pair_constraints: Dict[Tuple[str, str], List[BinaryRelation]] = {}
        self.neighbors: Dict[str, Set[str]] = {var: set() for var in self.variables}

    def add_binary_constraint(self, var1: str, var2: str, relation: BinaryRelation) -> None:
        constraint = BinaryConstraint(var1, var2, relation)
        self.constraints_by_var[var1].append(constraint)
        self.constraints_by_var[var2].append(constraint)
        self.neighbors[var1].add(var2)
        self.neighbors[var2].add(var1)
        self.pair_constraints.setdefault((var1, var2), []).append(relation)
        self.pair_constraints.setdefault((var2, var1), []).append(
            lambda x, y, rel=relation: rel(y, x)
        )

    def copy(self) -> "CSP":
        new_csp = CSP(self.variables, {var: set(values) for var, values in self.domains.items()})
        for var, constraints in self.constraints_by_var.items():
            for constraint in constraints:
                if constraint.var1 == var:
                    new_csp.add_binary_constraint(constraint.var1, constraint.var2, constraint.relation)
        return new_csp

    def domain_values(self, var: str) -> List[int]:
        return sorted(self.domains[var])

    def is_consistent(self, var: str, value: int, assignment: Dict[str, int]) -> bool:
        for constraint in self.constraints_by_var[var]:
            if not constraint.is_satisfied_with(var, value, assignment):
                return False
        return True

    def check_constraints(self, assignment: Dict[str, int]) -> bool:
        return all(
            all(constraint.is_satisfied(assignment) for constraint in constraints)
            for constraints in self.constraints_by_var.values()
        )

    def conflicts(self, var: str, value: int, assignment: Dict[str, int]) -> int:
        count = 0
        for constraint in self.constraints_by_var[var]:
            if not constraint.is_satisfied_with(var, value, assignment):
                count += 1
        return count

    def arcs(self) -> Iterable[Tuple[str, str]]:
        for var, neighbors in self.neighbors.items():
            for neighbor in neighbors:
                yield (var, neighbor)


@dataclass
class SearchStats:
    solution: Optional[Dict[str, int]]
    runtime: float
    nodes: int


def first_unassigned_variable(csp: CSP, assignment: Dict[str, int]) -> str:
    for var in csp.variables:
        if var not in assignment:
            return var
    raise ValueError("Alle Variablen bereits zugewiesen.")


def select_variable_mrv_degree(csp: CSP, assignment: Dict[str, int]) -> str:
    unassigned = [v for v in csp.variables if v not in assignment]

    def legal_values(var: str) -> List[int]:
        return [value for value in csp.domain_values(var) if csp.is_consistent(var, value, assignment)]

    legal_counts = {var: len(legal_values(var)) for var in unassigned}
    min_count = min(legal_counts.values())
    mrv_candidates = [var for var, count in legal_counts.items() if count == min_count]
    if len(mrv_candidates) == 1:
        return mrv_candidates[0]

    def degree(var: str) -> int:
        return sum(1 for neighbor in csp.neighbors[var] if neighbor not in assignment)

    return max(mrv_candidates, key=degree)


def backtracking_search(
    csp: CSP, variable_selector: Callable[[CSP, Dict[str, int]], str]
) -> SearchStats:
    nodes = 0

    def backtrack(assignment: Dict[str, int]) -> Optional[Dict[str, int]]:
        nonlocal nodes
        if len(assignment) == len(csp.variables):
            return assignment.copy()
        var = variable_selector(csp, assignment)
        for value in csp.domain_values(var):
            if csp.is_consistent(var, value, assignment):
                assignment[var] = value
                nodes += 1
                result = backtrack(assignment)
                if result is not None:
                    return result
                del assignment[var]
        return None

    start = time.perf_counter()
    solution = backtrack({})
    runtime = time.perf_counter() - start
    return SearchStats(solution=solution, runtime=runtime, nodes=nodes)


def bt_search(csp: CSP) -> SearchStats:
    return backtracking_search(csp, first_unassigned_variable)


def bt_search_mrv_degree(csp: CSP) -> SearchStats:
    return backtracking_search(csp, select_variable_mrv_degree)


def ac3(csp: CSP) -> bool:
    queue = deque(csp.arcs())
    while queue:
        xi, xj = queue.popleft()
        if revise(csp, xi, xj):
            if not csp.domains[xi]:
                return False
            for xk in csp.neighbors[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True


def revise(csp: CSP, xi: str, xj: str) -> bool:
    revised = False
    relations = csp.pair_constraints.get((xi, xj), [])
    if not relations:
        return revised

    to_remove: Set[int] = set()
    for x in list(csp.domains[xi]):
        supported = False
        for y in csp.domains[xj]:
            if all(rel(x, y) for rel in relations):
                supported = True
                break
        if not supported:
            to_remove.add(x)

    if to_remove:
        csp.domains[xi] -= to_remove
        revised = True
    return revised


def min_conflicts(
    csp: CSP, max_steps: int = 10000, max_restarts: int = 10, seed: Optional[int] = None
) -> SearchStats:
    rng = random.Random(seed)
    steps_total = 0
    start_overall = time.perf_counter()

    for _ in range(max_restarts):
        assignment: Dict[str, int] = {
            var: rng.choice(tuple(csp.domains[var])) for var in csp.variables
        }
        for step in range(max_steps):
            conflicted_vars = [
                var for var in csp.variables if csp.conflicts(var, assignment[var], assignment) > 0
            ]
            if not conflicted_vars:
                runtime = time.perf_counter() - start_overall
                return SearchStats(
                    solution=assignment.copy(), runtime=runtime, nodes=steps_total + step + 1
                )

            var = rng.choice(conflicted_vars)
            min_conflict = None
            best_values: List[int] = []
            for value in csp.domain_values(var):
                conflicts = csp.conflicts(var, value, assignment)
                if min_conflict is None or conflicts < min_conflict:
                    min_conflict = conflicts
                    best_values = [value]
                elif conflicts == min_conflict:
                    best_values.append(value)
            assignment[var] = rng.choice(best_values)
        steps_total += max_steps

    runtime = time.perf_counter() - start_overall
    return SearchStats(solution=None, runtime=runtime, nodes=steps_total)


def count_conflicts(csp: CSP, assignment: Dict[str, int]) -> int:
    conflicts = 0
    seen_pairs: Set[Tuple[str, str]] = set()
    for var, constraints in csp.constraints_by_var.items():
        for constraint in constraints:
            pair = tuple(sorted((constraint.var1, constraint.var2)))
            if pair in seen_pairs:
                continue
            if not constraint.is_satisfied(assignment):
                conflicts += 1
            seen_pairs.add(pair)
    return conflicts


def build_zebra_csp() -> CSP:
    houses = {1, 2, 3, 4, 5}

    categories = {
        "Farben": ["gelb_haus", "blau_haus", "rot_haus", "weiß_haus", "grün_haus"],
        "Nationalität": [
            "norweger_haus",
            "ukrainer_haus",
            "engländer_haus",
            "spanier_haus",
            "japaner_haus",
        ],
        "Haustier": ["fuchs_haus", "pferd_haus", "schnecken_haus", "hund_haus", "zebra_haus"],
        "Getränk": ["wasser_haus", "tee_haus", "milch_haus", "osaft_haus", "kaffee_haus"],
        "Zigaretten": [
            "kools_haus",
            "chesterfield_haus",
            "oldgold_haus",
            "luckystrike_haus",
            "parliament_haus",
        ],
    }

    variables = list(itertools.chain.from_iterable(categories.values()))
    domains = {var: houses.copy() for var in variables}

    domains["milch_haus"] = {3}
    domains["norweger_haus"] = {1}

    csp = CSP(variables, domains)

    for group in categories.values():
        for var1, var2 in itertools.combinations(group, 2):
            csp.add_binary_constraint(var1, var2, lambda x, y: x != y)

    equal_pairs = [
        ("engländer_haus", "rot_haus"),
        ("spanier_haus", "hund_haus"),
        ("kaffee_haus", "grün_haus"),
        ("ukrainer_haus", "tee_haus"),
        ("oldgold_haus", "schnecken_haus"),
        ("kools_haus", "gelb_haus"),
        ("luckystrike_haus", "osaft_haus"),
        ("japaner_haus", "parliament_haus"),
    ]
    for var1, var2 in equal_pairs:
        csp.add_binary_constraint(var1, var2, lambda x, y: x == y)

    csp.add_binary_constraint("grün_haus", "weiß_haus", lambda g, w: g == w + 1)
    csp.add_binary_constraint("chesterfield_haus", "fuchs_haus", lambda c, f: abs(c - f) == 1)
    csp.add_binary_constraint("kools_haus", "pferd_haus", lambda k, p: abs(k - p) == 1)
    csp.add_binary_constraint("norweger_haus", "blau_haus", lambda n, b: abs(n - b) == 1)

    return csp


def print_solution(solution: Dict[str, int]) -> None:
    houses = {i: {} for i in range(1, 6)}
    for var, house in solution.items():
        attribute, _ = var.split("_", 1)
        houses[house][attribute] = var.replace("_haus", "")

    print("Lösung (Hausnummer -> Eigenschaften):")
    for house in range(1, 6):
        attrs = houses[house]
        sorted_attrs = ", ".join(sorted(attrs.values()))
        print(f"  Haus {house}: {sorted_attrs}")


def run_experiments() -> None:
    random.seed(42)
    experiments = []

    csp = build_zebra_csp()
    stats_bt = bt_search(csp.copy())
    experiments.append(("Backtracking", stats_bt))

    stats_bt_mrv = bt_search_mrv_degree(csp.copy())
    experiments.append(("Backtracking (MRV+Grad)", stats_bt_mrv))

    csp_ac3 = csp.copy()
    start = time.perf_counter()
    ac3_result = ac3(csp_ac3)
    ac3_time = time.perf_counter() - start
    ac3_solution = None
    if not ac3_result:
        experiments.append(("AC-3", SearchStats(solution=None, runtime=ac3_time, nodes=0)))
    elif all(len(domain) == 1 for domain in csp_ac3.domains.values()):
        ac3_solution = {var: next(iter(values)) for var, values in csp_ac3.domains.items()}
        experiments.append(("AC-3", SearchStats(solution=ac3_solution, runtime=ac3_time, nodes=0)))
    else:
        experiments.append(("AC-3 (Präprozessierung)", SearchStats(solution=None, runtime=ac3_time, nodes=0)))
        stats_after_ac3 = bt_search_mrv_degree(csp_ac3)
        stats_after_ac3 = SearchStats(
            solution=stats_after_ac3.solution,
            runtime=stats_after_ac3.runtime + ac3_time,
            nodes=stats_after_ac3.nodes,
        )
        experiments.append(("AC-3 + Backtracking (MRV+Grad)", stats_after_ac3))

    stats_min_conflicts = min_conflicts(csp.copy(), max_steps=5000, seed=42)
    experiments.append(("Min-Conflicts", stats_min_conflicts))

    print("\n--- Vergleich der Verfahren ---")
    for name, stats in experiments:
        solution_status = "gefunden" if stats.solution else "keine Lösung"
        print(
            f"{name:30s} -> Lösung: {solution_status:12s} | "
            f"Laufzeit: {stats.runtime:.4f} s | Knoten/Steps: {stats.nodes}"
        )

    final_solution = (
        ac3_solution
        or stats_bt_mrv.solution
        or stats_bt.solution
        or stats_min_conflicts.solution
    )
    if final_solution:
        print()
        print_solution(final_solution)
    else:
        print("\nKeine Lösung gefunden.")


if __name__ == "__main__":
    run_experiments()
