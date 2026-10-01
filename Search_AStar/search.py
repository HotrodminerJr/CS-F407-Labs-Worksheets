"""Shared engine for the Search / A* lab.

The lab models the warehouse as a search problem P = (S, A, T, s0, G, c):

    S     : grid cells (row, col) that are not walls
    A     : {Up, Down, Left, Right}
    T     : (r,c) + delta, provided the destination is in S
    s0    : the single cell containing 'S' on the map
    G     : the single cell containing 'G' on the map
    c     : uniform cost 1 per move

Both BFS and A* are implemented in the same style so their state counts can
be compared fairly. Each returns a `SearchResult` carrying:
    - path (list of (r, c)) or None
    - path_cost
    - states_expanded  (number of nodes popped from the frontier)
    - frontier_pushes  (just for curiosity)
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

Pos = Tuple[int, int]

# Up, Down, Left, Right — kept in the lab's exact order so printed traces match
MOVES: List[Tuple[str, Tuple[int, int]]] = [
    ("Up",    (-1,  0)),
    ("Down",  ( 1,  0)),
    ("Left",  ( 0, -1)),
    ("Right", ( 0,  1)),
]


# ---------- Grid -----------------------------------------------------

@dataclass
class Grid:
    cells: List[str]                 # one string per row
    start: Pos
    goal: Pos

    @property
    def rows(self) -> int:
        return len(self.cells)

    @property
    def cols(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    def passable(self, p: Pos) -> bool:
        r, c = p
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return self.cells[r][c] != "#"

    def neighbours(self, p: Pos):
        r, c = p
        for name, (dr, dc) in MOVES:
            q = (r + dr, c + dc)
            if self.passable(q):
                yield name, q


def parse_grid(text: str) -> Grid:
    """Parse an ASCII map. Rows are padded with ' ' only if required to stay rectangular."""
    rows = [line.rstrip("\n") for line in text.strip("\n").splitlines() if line.strip()]
    width = max(len(r) for r in rows)
    rows = [r.ljust(width, "#") for r in rows]  # treat trailing missing cells as walls
    start = goal = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    assert start is not None, "map has no 'S' cell"
    assert goal is not None, "map has no 'G' cell"
    return Grid(cells=rows, start=start, goal=goal)


def render(grid: Grid, path: Optional[List[Pos]] = None) -> str:
    overlay = [list(row) for row in grid.cells]
    if path:
        for (r, c) in path:
            if overlay[r][c] in (".", " "):
                overlay[r][c] = "*"
    overlay[grid.start[0]][grid.start[1]] = "S"
    overlay[grid.goal[0]][grid.goal[1]] = "G"
    return "\n".join("".join(row) for row in overlay)


# ---------- Heuristics ----------------------------------------------

def h_zero(_: Pos, __: Pos) -> float:
    return 0.0


def h_manhattan(p: Pos, g: Pos) -> float:
    return abs(p[0] - g[0]) + abs(p[1] - g[1])


def h_euclidean(p: Pos, g: Pos) -> float:
    return math.hypot(p[0] - g[0], p[1] - g[1])


def h_manhattan_times_2(p: Pos, g: Pos) -> float:
    # Deliberately inadmissible: h > h* on the 4-connected grid.
    return 2.0 * h_manhattan(p, g)


# ---------- Search --------------------------------------------------

@dataclass
class SearchResult:
    path: Optional[List[Pos]]
    path_cost: int
    states_expanded: int
    frontier_pushes: int

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def path_length(self) -> int:
        return 0 if self.path is None else max(0, len(self.path) - 1)


def _reconstruct(parent: dict, end: Pos) -> List[Pos]:
    path = [end]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def bfs(grid: Grid) -> SearchResult:
    s0, g = grid.start, grid.goal
    if s0 == g:
        return SearchResult(path=[s0], path_cost=0, states_expanded=0, frontier_pushes=1)

    frontier: deque[Pos] = deque([s0])
    parent = {s0: None}
    expanded = 0
    pushes = 1
    while frontier:
        n = frontier.popleft()
        expanded += 1
        if n == g:
            path = _reconstruct(parent, n)
            return SearchResult(path=path, path_cost=len(path) - 1,
                                states_expanded=expanded, frontier_pushes=pushes)
        for _, nb in grid.neighbours(n):
            if nb not in parent:
                parent[nb] = n
                frontier.append(nb)
                pushes += 1
    return SearchResult(path=None, path_cost=0, states_expanded=expanded, frontier_pushes=pushes)


def astar(grid: Grid, h: Callable[[Pos, Pos], float] = h_manhattan) -> SearchResult:
    s0, g = grid.start, grid.goal
    # frontier entries: (f, counter, state) — counter is a tie-breaker for stable order
    counter = 0
    frontier: List[Tuple[float, int, Pos]] = []
    heapq.heappush(frontier, (h(s0, g), counter, s0))
    counter += 1

    g_cost = {s0: 0}
    parent = {s0: None}
    closed = set()
    expanded = 0
    pushes = 1

    while frontier:
        f, _, n = heapq.heappop(frontier)
        if n in closed:
            continue
        closed.add(n)
        expanded += 1
        if n == g:
            path = _reconstruct(parent, n)
            return SearchResult(path=path, path_cost=g_cost[n],
                                states_expanded=expanded, frontier_pushes=pushes)
        for _, nb in grid.neighbours(n):
            if nb in closed:
                continue
            tentative = g_cost[n] + 1
            if nb not in g_cost or tentative < g_cost[nb]:
                g_cost[nb] = tentative
                parent[nb] = n
                heapq.heappush(frontier, (tentative + h(nb, g), counter, nb))
                counter += 1
                pushes += 1
    return SearchResult(path=None, path_cost=0, states_expanded=expanded, frontier_pushes=pushes)


# ---------- The lab's map ------------------------------------------

LAB_MAP = """\
################
#S....#........#
#.###.#.#######
#...#.#.......#
###.#.########.#
#...#.........#
#.###########.#
#.............#
################
"""

# A corner of the above has 'G'; the lab's exact map uses a 16-wide grid with G
# at the lower-right. We re-use `parse_grid` by placing G explicitly:
def lab_grid() -> Grid:
    text = LAB_MAP
    rows = text.strip("\n").splitlines()
    # Place G at the inner lower-right (second-last column, second-last row).
    r = len(rows) - 2
    row = list(rows[r])
    # Find rightmost '.' in that row and replace with G.
    for c in range(len(row) - 1, -1, -1):
        if row[c] == ".":
            row[c] = "G"
            break
    rows[r] = "".join(row)
    return parse_grid("\n".join(rows))
