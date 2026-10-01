"""Task 3 of the Search lab: four systematic tests of the A* implementation."""
from pathlib import Path

from search import astar, bfs, parse_grid, lab_grid, render, h_manhattan

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


def banner(txt: str) -> str:
    bar = "=" * len(txt)
    return f"\n{bar}\n{txt}\n{bar}"


def report(name: str, grid, result) -> str:
    out = [banner(name)]
    out.append(render(grid))
    out.append("")
    out.append(f"solution found : {result.found}")
    out.append(f"path length    : {result.path_length}")
    out.append(f"path cost      : {result.path_cost}")
    out.append(f"states expanded: {result.states_expanded}")
    if result.found:
        out.append("\npath overlay (* = path):")
        out.append(render(grid, result.path))
    return "\n".join(out)


# ------- Test 1: original warehouse -------
def test_1() -> str:
    g = lab_grid()
    r = astar(g, h_manhattan)
    return report("Test 1 - Original warehouse (A*, Manhattan)", g, r)


# ------- Test 2: trivial case -------
TRIVIAL = """\
#####
#SG##
#####
"""

def test_2() -> str:
    g = parse_grid(TRIVIAL)
    r = astar(g, h_manhattan)
    return report("Test 2 - Trivial case (S next to G)", g, r)


# ------- Test 3: no solution -------
NO_SOLN = """\
#######
#S....#
###.###
#...#G#
#######
"""

def test_3() -> str:
    g = parse_grid(NO_SOLN)
    r = astar(g, h_manhattan)
    out = [report("Test 3 - No solution (G walled off)", g, r)]
    if r.found:
        out.append("FAIL: planner should have reported no solution.")
    else:
        out.append("PASS: planner terminated with no path, not an infinite loop.")
    return "\n".join(out)


# ------- Test 4: alternative paths -------
# Two equally short routes around a thin wall. A* should still return a shortest path.
ALT_PATHS = """\
#########
#S......#
#.#####.#
#.......#
#.#####.#
#......G#
#########
"""

def test_4() -> str:
    g = parse_grid(ALT_PATHS)
    r = astar(g, h_manhattan)
    out = [report("Test 4 - Alternative paths (shortest should be returned)", g, r)]
    # independent lower bound: BFS gives the exact shortest-path length for unit cost
    rb = bfs(g)
    out.append(f"BFS shortest length (independent): {rb.path_length}")
    if r.found and r.path_length == rb.path_length:
        out.append(f"PASS: A* path length ({r.path_length}) == BFS optimum ({rb.path_length}).")
    else:
        out.append("FAIL: A* did not return a shortest path.")
    return "\n".join(out)


def main() -> None:
    sections = [test_1(), test_2(), test_3(), test_4()]
    text = "\n".join(sections) + "\n"
    print(text)
    (OUT / "test_results.log").write_text(text)
    print(f"\nsaved outputs/test_results.log")


if __name__ == "__main__":
    main()
