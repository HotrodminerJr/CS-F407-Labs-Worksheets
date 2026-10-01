# Lab — Agents: Goal-Based Agent for Warehouse Navigation

**Course:** CS F407 — Artificial Intelligence
**Worksheet:** [`agents_lab.pdf`](agents_lab.pdf)

**Files:**

| File | Role |
|---|---|
| [`agent.py`](agent.py) | `Grid`, BFS search, `GoalBasedAgent` |
| [`run.py`](run.py) | Executes the agent on the warehouse map, saves `outputs/run.log` |
| [`outputs/run.log`](outputs/run.log) | Captured output |

**Reproduce:**

```bash
python run.py
```

Pure stdlib.

---

## Task 1 — Understanding the problem

1. **Environment.** A static 2D grid (`list[str]`) with four kinds of cells:
   `#` (wall), `.` (free), `S` (start), `G` (goal). The lab's map is a 7-row
   warehouse with shelving blocks between `S` (upper-left corner) and `G`
   (upper-right corner).
2. **Goal.** Reach the single cell containing `G` from `S` without crossing any
   `#`.
3. **Available actions.** `{Up, Down, Left, Right}`, each moving the agent by
   one cell in that direction, provided the destination is in bounds and not a
   wall.
4. **Information the agent must maintain.** Its current position, the goal
   position, the warehouse map (walls), and either the plan it is currently
   executing or at least a search frontier. Without the map it cannot tell
   which actions are legal; without the goal it cannot distinguish success
   from any other state.
5. **Why this is goal-based, not reflex.** A reflex agent chooses an action
   from the current percept alone — e.g. "if G is east of me, go east". That
   fails the first time a wall sits between the agent and the goal: the reflex
   rule says "east" but the correct action is "go south first, around the
   shelf". A goal-based agent commits to the *objective* `reach(G)` and uses
   search to produce a sequence of actions that provably achieves it.

**Scaling consideration.** If the warehouse were twice as large, BFS still
finds the optimal plan, but the number of states expanded grows roughly with
the free-cell count. The limit is memory for the frontier/visited set rather
than CPU time at this scale; for very large maps you would move to A* with a
Manhattan-distance heuristic (see the companion [Search_AStar](../Search_AStar/)
lab, where A* expanded 25 vs BFS's 37 on the same shape of problem).

---

## Task 2 — Agent design

### Block diagram

```
            +------------------+
            |   Warehouse      |
            |   (grid)         |
            +--------+---------+
                     |
                     | percept: current (row, col)
                     v
            +--------+---------+
            |  Agent memory    |
            |  - current state |
            |  - goal state    |
            |  - plan (deque)  |
            +--------+---------+
                     |
                     v
            +------------------+
            | Decision-making  |
            |  if plan empty:  |
            |    plan = BFS(   |
            |      state, goal)|
            |  pop next action |
            +--------+---------+
                     |
                     | action in {Up, Down, Left, Right}
                     v
            +------------------+
            |   Warehouse      |
            |   (grid)         |
            +------------------+
```

### Component specification

| Component | Specification |
|---|---|
| Environment       | Immutable `Grid` object (walls, free cells, S, G) |
| Current state     | `(row, col)` tuple |
| Goal              | `(row, col)` tuple |
| Available actions | `{Up, Down, Left, Right}` filtered through `Grid.passable` |
| Decision-making   | BFS over grid positions; cache plan on the agent; emit one action per step |

---

## Task 3 — Prompt engineering + results

### Prompt used

> *I am writing a goal-based agent in Python for a 2D warehouse navigation
> problem. The warehouse is an ASCII grid where `#` is a wall, `.` is free
> space, `S` is the start, and `G` is the goal. The agent can move up, down,
> left, or right one cell at a time, and cannot cross `#`. Produce a
> stdlib-only program with (i) a `Grid` dataclass that parses the ASCII map
> and exposes a `passable` and `neighbours` interface; (ii) a BFS function
> that returns the sequence of actions and the state trace from `S` to `G`,
> or `None` if no path exists; (iii) a `GoalBasedAgent` class that plans
> once with BFS, caches the plan, and emits one action per `choose_action()`
> call. The program should also print the warehouse with the path overlaid
> and report the number of states expanded.*

### Observations after running

Captured in [`outputs/run.log`](outputs/run.log).

```
Search algorithm      : BFS over 4-connected grid cells
States expanded       : 52
Plan length (actions) : 20
Start                 : (1, 1)
Goal                  : (1, 19)

Action sequence:
  Right -> Right -> Right -> Down -> Right -> Right -> Right -> Up ->
  Right -> Right -> Right -> Right -> Right -> Right -> Right -> Right ->
  Right -> Right -> Right -> Right
```

Final map with path overlaid:

```
#####################
#S***.#************G#
#.##****##########..#
#.....##............#
#.######.###.#.###..#
#........#..........#
#####################
```

### The four questions

1. **Did the LLM generate a working program on the first attempt?** Mostly.
   The core BFS and `Grid` were right; the first draft forgot to detect
   `start == goal` as a zero-action solution, and used a plain `list` as the
   frontier (which turns BFS into DFS). Both were fixed before accepting.
2. **How could I improve the prompt?** Explicitly ask for *(a)* the
   `s0 == g` edge case and *(b)* `collections.deque` as the frontier so the
   `popleft` is O(1). Also require the program to report both the action
   sequence and a visualisation — if you only ask for one, you may get a
   silently-wrong plan that reads reasonably.
3. **What search algorithm did the LLM choose?** BFS. On a uniform-cost grid
   that is a sound default — it always returns an optimal path and is easy to
   verify (unlike DFS, which may return a non-shortest path, or A*, which
   requires picking an admissible heuristic).
4. **Why did it choose BFS?** Because the problem has uniform step costs and a
   finite, small state space, so BFS is the simplest search algorithm that is
   both complete and optimal. For a *larger* warehouse the LLM would almost
   certainly suggest A* with Manhattan distance — exactly the extension the
   Search / A* lab studies.
