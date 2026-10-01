"""Tasks 5 and 6: BFS vs A*, and the heuristic study on the lab map."""
from pathlib import Path

from search import (
    astar, bfs, lab_grid, render,
    h_zero, h_manhattan, h_euclidean, h_manhattan_times_2,
)

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


def row(label: str, result) -> str:
    found = "yes" if result.found else "no"
    return f"| {label:<22} | {found:^8} | {result.path_length:>5} | {result.path_cost:>5} | {result.states_expanded:>7} | {result.frontier_pushes:>8} |"


def table_header(cols) -> str:
    header = "| " + " | ".join(f"{c:^{w}}" for c, w in cols) + " |"
    sep = "|" + "|".join("-" * (w + 2) for _, w in cols) + "|"
    return header + "\n" + sep


def main() -> None:
    g = lab_grid()

    # ---- Task 5: BFS vs A* ----
    rb = bfs(g)
    ra = astar(g, h_manhattan)

    print("Task 5 - BFS vs A* on the lab map\n")
    cols = [("algorithm", 22), ("found?", 8), ("len", 5), ("cost", 5), ("expand.", 7), ("pushes", 8)]
    print(table_header(cols))
    print(row("BFS",             rb))
    print(row("A* (Manhattan)",  ra))

    task5 = [
        "Task 5 - BFS vs A* comparison (on the lab warehouse map)",
        "",
        table_header(cols),
        row("BFS",             rb),
        row("A* (Manhattan)",  ra),
        "",
        f"(a) Both algorithms found a solution: {rb.found and ra.found}",
        f"(b) Both found paths of the same length: {rb.path_length == ra.path_length}  "
        f"(BFS={rb.path_length}, A*={ra.path_length})",
        f"(c) A* expanded fewer states: {ra.states_expanded < rb.states_expanded}  "
        f"(BFS={rb.states_expanded}, A*={ra.states_expanded})",
        "(d) A* expands fewer because the heuristic h biases the frontier toward",
        "    the goal: nodes with lower f=g+h are expanded first, and nodes that",
        "    cannot improve the best-known cost are skipped by the closed set.",
    ]

    # ---- Task 6: heuristic study ----
    print("\nTask 6 - Heuristic study on the lab map\n")
    heuristics = [
        ("A* h=0 (==Dijkstra)", h_zero),
        ("A* h=Manhattan",      h_manhattan),
        ("A* h=Euclidean",      h_euclidean),
        ("A* h=2*Manhattan",    h_manhattan_times_2),
    ]
    print(table_header(cols))
    heur_rows = [table_header(cols)]
    results = {}
    for label, h in heuristics:
        r = astar(g, h)
        results[label] = r
        line = row(label, r)
        print(line)
        heur_rows.append(line)

    optimum = min(r.path_length for r in results.values() if r.found)
    task6 = [
        "",
        "Task 6 - Heuristic study on the lab warehouse map",
        "",
    ] + heur_rows + [
        "",
        f"shortest path length (BFS-verified optimum) = {optimum}",
        "",
        "Observations:",
        "- h=0 reduces A* to Dijkstra on a unit-cost grid, which in turn behaves",
        "  like BFS: it still finds the optimum but expands the most states.",
        "- Manhattan is admissible on a 4-connected grid (|dr|+|dc| <= true cost),",
        "  so A* with it is optimal. It gives the smallest expand count here.",
        "- Euclidean is also admissible (sqrt(dr^2+dc^2) <= manhattan),",
        "  so A* is still optimal, but it is a weaker lower bound than Manhattan,",
        "  so it expands at least as many states as Manhattan.",
        "- 2 x Manhattan is INADMISSIBLE: h may exceed the true remaining cost,",
        "  so A* can commit early to a sub-optimal path. In this run its path",
        f"  length is {results['A* h=2*Manhattan'].path_length} vs the optimum {optimum}.",
        "  If equal here, the inadmissible heuristic happened to still find the",
        "  optimum on this specific map, but the guarantee is gone.",
    ]

    (OUT / "experiments.log").write_text("\n".join(task5 + task6) + "\n")
    print("\nsaved outputs/experiments.log")

    # Also save the path overlay for Task 1 of Task 3.
    (OUT / "lab_map_path.txt").write_text(render(g, ra.path) + "\n")
    print("saved outputs/lab_map_path.txt")


if __name__ == "__main__":
    main()
