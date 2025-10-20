from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Dict, List, Sequence, Tuple

# Ortskoordinaten (aus der Aufgabenkarte extrahiert)
COORDS: Dict[str, Tuple[float, float]] = {
    "Frankfurt": (156.23683, 21.794117),
    "Mannheim": (61.736824, 81.441177),
    "Würzburg": (156.23683, 115.94118),
    "Karlsruhe": (56.502716, 193.23529),
    "Stuttgart": (259.27957, 97.941177),
    "Nürnberg": (235.54427, 193.23529),
    "Erfurt": (145.50392, 211.23529),
    "Augsburg": (128.00154, 270.88239),
    "München": (225.06036, 348.52942),
    "Kassel": (320.99631, 139.58824),
}

# Straßennetz mit Kantengewichten (Kilometer)
EDGES: Sequence[Tuple[str, str, int]] = (
    ("Frankfurt", "Mannheim", 85),
    ("Frankfurt", "Würzburg", 217),
    ("Frankfurt", "Kassel", 173),
    ("Mannheim", "Karlsruhe", 80),
    ("Stuttgart", "Nürnberg", 183),
    ("Würzburg", "Nürnberg", 103),
    ("Würzburg", "Erfurt", 186),
    ("Karlsruhe", "Augsburg", 250),
    ("Nürnberg", "München", 167),
    ("Augsburg", "München", 84),
    ("Kassel", "München", 502),
)


def build_graph() -> Dict[str, Dict[str, int]]:
    graph: Dict[str, Dict[str, int]] = {node: {} for node in COORDS}
    for a, b, cost in EDGES:
        graph[a][b] = cost
        graph[b][a] = cost
    return graph


def path_cost(path: Sequence[str], graph: Dict[str, Dict[str, int]]) -> int:
    return sum(graph[a][b] for a, b in zip(path, path[1:]))


def depth_first_graph_search(start: str, goal: str, graph: Dict[str, Dict[str, int]]) -> Tuple[List[str], int, int, int]:
    stack: List[Tuple[str, List[str]]] = [(start, [start])]
    visited = set()
    iterations = 0
    max_frontier = len(stack)
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        iterations += 1
        if node == goal:
            return path, path_cost(path, graph), iterations, max_frontier
        neighbors = sorted(graph[node])
        for neighbor in reversed(neighbors):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
        max_frontier = max(max_frontier, len(stack))
    raise ValueError("Kein Pfad gefunden.")


def breadth_first_graph_search(start: str, goal: str, graph: Dict[str, Dict[str, int]]) -> Tuple[List[str], int, int, int]:
    queue: deque[Tuple[str, List[str]]] = deque([(start, [start])])
    visited = {start}
    iterations = 0
    max_frontier = len(queue)
    while queue:
        node, path = queue.popleft()
        iterations += 1
        if node == goal:
            return path, path_cost(path, graph), iterations, max_frontier
        for neighbor in sorted(graph[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        max_frontier = max(max_frontier, len(queue))
    raise ValueError("Kein Pfad gefunden.")


def euclidean_heuristics(scale: float = 1.0) -> Dict[str, float]:
    goal_x, goal_y = COORDS["München"]
    heuristics = {}
    for city, (x, y) in COORDS.items():
        if city == "München":
            heuristics[city] = 0.0
        else:
            heuristics[city] = math.hypot(goal_x - x, goal_y - y) * scale
    return heuristics


def shortest_paths_from(goal: str, graph: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    distances = {node: math.inf for node in graph}
    distances[goal] = 0
    queue: List[Tuple[int, str]] = [(0, goal)]
    while queue:
        dist, node = heapq.heappop(queue)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node].items():
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))
    return distances


def admissible_heuristics(graph: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    raw = euclidean_heuristics(scale=1.0)
    shortest = shortest_paths_from("München", graph)
    scale = min(
        shortest[city] / raw_val
        for city, raw_val in raw.items()
        if city != "München" and raw_val > 0
    )
    return euclidean_heuristics(scale=scale)


def astar_tree_search(start: str, goal: str, graph: Dict[str, Dict[str, int]], heuristics: Dict[str, float]) -> Tuple[List[str], int, int, int]:
    counter = 0
    frontier: List[Tuple[float, int, str, List[str], int]] = [
        (heuristics[start], counter, start, [start], 0)
    ]
    iterations = 0
    max_frontier = len(frontier)
    while frontier:
        f_val, _, node, path, g_val = heapq.heappop(frontier)
        iterations += 1
        if node == goal:
            return path, g_val, iterations, max_frontier
        for neighbor in sorted(graph[node]):
            if neighbor in path:
                continue
            new_g = g_val + graph[node][neighbor]
            counter += 1
            heapq.heappush(
                frontier,
                (new_g + heuristics[neighbor], counter, neighbor, path + [neighbor], new_g),
            )
        max_frontier = max(max_frontier, len(frontier))
    raise ValueError("Kein Pfad gefunden.")


def main() -> None:
    graph = build_graph()
    heuristics = admissible_heuristics(graph)

    dfs_path, dfs_cost, dfs_iters, dfs_frontier = depth_first_graph_search("Würzburg", "München", graph)
    bfs_path, bfs_cost, bfs_iters, bfs_frontier = breadth_first_graph_search("Würzburg", "München", graph)
    astar_path, astar_cost, astar_iters, astar_frontier = astar_tree_search("Würzburg", "München", graph, heuristics)

    print("Suchvergleich für Würzburg → München")
    print(f"DFS (Graph-Search):   Pfad {dfs_path}, Kosten {dfs_cost} km, Iterationen {dfs_iters}, max. Stackgröße {dfs_frontier}")
    print(f"BFS (Graph-Search):   Pfad {bfs_path}, Kosten {bfs_cost} km, Iterationen {bfs_iters}, max. Queuegröße {bfs_frontier}")
    print(f"A* (Tree-Search):     Pfad {astar_path}, Kosten {astar_cost} km, Iterationen {astar_iters}, max. Frontiergröße {astar_frontier}")


if __name__ == "__main__":
    main()

