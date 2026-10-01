"""Goal-based agent for the warehouse navigation problem.

Design (Task 2 of the lab):

    +-----------+       percept      +------------------+
    | Warehouse |-------------------->|  Agent memory    |
    | (grid)    |                     |  - current state |
    +-----^-----+                     |  - goal          |
          |          action           |  - plan (deque)  |
          +---------------------------|                  |
                                      +---------+--------+
                                                |
                                                v
                                   +---------------------------+
                                   |  Decision-making:         |
                                   |   if plan empty:          |
                                   |     plan = BFS(state,goal)|
                                   |   action = plan.popleft() |
                                   +---------------------------+

This is a *goal-based* agent, not a reflex agent: it searches for a sequence
of actions achieving the goal (BFS over grid positions, since all moves cost
the same) and then executes the plan step-by-step. A reflex agent cannot do
this — it would need a rule like "if I see G to my east, go east", which
breaks the first time the goal is behind a wall.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

Pos = Tuple[int, int]

# The warehouse map from the lab (exactly as printed in the PDF).
WAREHOUSE = """\
####################
#S....#............G#
#.##....##########..#
#.....##............#
#.######.###.#.###..#
#........#..........#
####################
"""

MOVES = [
    ("Up",    (-1,  0)),
    ("Down",  ( 1,  0)),
    ("Left",  ( 0, -1)),
    ("Right", ( 0,  1)),
]


@dataclass
class Grid:
    cells: List[str]
    start: Pos
    goal: Pos

    @property
    def rows(self) -> int:
        return len(self.cells)

    @property
    def cols(self) -> int:
        return len(self.cells[0])

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


def parse(text: str = WAREHOUSE) -> Grid:
    rows = [line for line in text.strip("\n").splitlines() if line.strip()]
    width = max(len(r) for r in rows)
    rows = [r.ljust(width, "#") for r in rows]
    start = goal = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    assert start and goal, "map must contain both S and G"
    return Grid(cells=rows, start=start, goal=goal)


# ---- Search component (BFS) ----
def bfs(grid: Grid) -> Tuple[Optional[List[str]], Optional[List[Pos]], int]:
    """Return (action_sequence, state_trace, states_expanded)."""
    s0, g = grid.start, grid.goal
    if s0 == g:
        return [], [s0], 0

    frontier: deque[Pos] = deque([s0])
    parent: dict[Pos, Tuple[Pos, str] | None] = {s0: None}
    expanded = 0
    while frontier:
        n = frontier.popleft()
        expanded += 1
        if n == g:
            acts, trace = _reconstruct(parent, n)
            return acts, trace, expanded
        for name, nb in grid.neighbours(n):
            if nb not in parent:
                parent[nb] = (n, name)
                frontier.append(nb)
    return None, None, expanded


def _reconstruct(parent, end: Pos):
    actions_rev: List[str] = []
    trace_rev: List[Pos] = [end]
    cur = end
    while parent[cur] is not None:
        prev, name = parent[cur]
        actions_rev.append(name)
        trace_rev.append(prev)
        cur = prev
    return list(reversed(actions_rev)), list(reversed(trace_rev))


# ---- Goal-based agent (plan then execute) ----
@dataclass
class GoalBasedAgent:
    grid: Grid
    current: Pos = field(init=False)
    goal: Pos = field(init=False)
    plan: Deque[str] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.current = self.grid.start
        self.goal = self.grid.goal

    def perceive(self, position: Pos) -> None:
        self.current = position

    def choose_action(self) -> Optional[str]:
        if self.current == self.goal:
            return None
        if not self.plan:
            # The lab asks for one planning pass. We cache it on the agent.
            acts, _, _ = bfs(self.grid)
            if acts is None:
                return None
            self.plan.extend(acts)
        return self.plan.popleft() if self.plan else None


# ---- Rendering ----
def render(grid: Grid, trace: Optional[List[Pos]] = None) -> str:
    overlay = [list(r) for r in grid.cells]
    if trace:
        for (r, c) in trace:
            if overlay[r][c] == ".":
                overlay[r][c] = "*"
    overlay[grid.start[0]][grid.start[1]] = "S"
    overlay[grid.goal[0]][grid.goal[1]] = "G"
    return "\n".join("".join(row) for row in overlay)
