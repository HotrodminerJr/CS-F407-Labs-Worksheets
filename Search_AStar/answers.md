# Lab — Search and A*

**Course:** CS F407 — Artificial Intelligence
**Worksheet:** [`search_lab_ex.pdf`](search_lab_ex.pdf)

**Code in this folder:**

| File | Role |
|---|---|
| [`search.py`](search.py) | Grid parser, BFS, A* with pluggable heuristic, 4 heuristics |
| [`tests.py`](tests.py) | Task 3 — four systematic tests of the implementation |
| [`experiments.py`](experiments.py) | Task 5 (BFS vs A*) and Task 6 (heuristic study) |
| [`outputs/`](outputs/) | Captured logs and the solved-map overlay |

**Reproduce:**

```bash
python tests.py
python experiments.py
```

Pure stdlib, no non-standard dependencies.

---

## Task 0 — The warehouse as a search problem

| Component | Specification |
|---|---|
| State **S** | Grid cell `(r, c)` with `0 ≤ r < rows`, `0 ≤ c < cols` such that `cells[r][c] ≠ '#'` |
| Actions **A** | `{Up, Down, Left, Right}` |
| Transition **T** | `T((r,c), Up) = (r−1, c)` etc., provided the destination is in `S`; otherwise undefined |
| Initial state **s₀** | The unique cell containing `'S'` on the map |
| Goal **G** | `{ cell containing 'G' }` (singleton) |
| Cost **c** | 1 per move (uniform) |

**(a) Information required to specify a state.** Only the agent's `(row, col)`
position. The grid itself is static, so it is part of the problem, not part of
the state.

**(b) What makes an action invalid?** The target cell is outside the grid, or
equals `'#'`. Both are rejected by `Grid.passable` and never enter the frontier.

**(c) Is this deterministic?** Yes — `T` is a (partial) function. Every state /
action pair has at most one successor.

**(d) What is a solution?** A finite sequence of actions whose transitions
starting from `s₀` end in a state belonging to `G`. For this lab we report the
**sequence of states** (easier to visualise on the map) and the number of moves
= path length = path cost (since every action costs 1).

---

## Task 1 — Agent design (before any code)

| Decision | Choice |
|---|---|
| State in Python | `tuple[int, int]` — immutable, hashable, cheap in a `set`/`dict` |
| Warehouse | `list[str]` (one string per row); walls checked by character comparison |
| Valid actions | For each state, iterate `{Up, Down, Left, Right}` and keep those whose destination `Grid.passable` is `True` |
| Goal test | `n == grid.goal` |
| Frontier for A* | min-heap of `(f, counter, state)` with `counter` as a tie-breaker |
| Path reconstruction | `parent: dict[Pos, Optional[Pos]]`, follow pointers from goal back to start |
| Report | `found?`, path, path length, path cost, `states_expanded`, `frontier_pushes` |

---

## Task 2 — Prompt used for the LLM

> *I am implementing a goal-based A\* search agent in Python. The environment
> is a 2D grid with `#` as walls, `.` as free cells, `S` as start, `G` as goal.
> Represent each state as `(row, col)`. The agent can move up, down, left, or
> right one cell at a time; every move costs 1. Implement A\* using a min-heap
> frontier keyed on `f(n) = g(n) + h(n)`. Use Manhattan distance as the
> heuristic. Maintain a closed set so no state is expanded twice. On
> termination, report whether a solution was found, the reconstructed path,
> the path length, and the number of states expanded. Also implement BFS with
> the same reporting interface so the two can be compared on the same map.
> Keep the implementation library-free; only `collections` and `heapq` are
> allowed. Expose heuristics as pluggable functions so I can swap in
> `h = 0`, Euclidean, or `2 × Manhattan`.*

**Changes I made before accepting the generated code:**

1. Added a **`counter` tie-breaker** in the heap. Without one, Python's
   heap tries to compare `Pos` tuples when `f` values collide, which it can
   do fine — but if I ever changed states to a type that is not totally
   ordered, the heap would raise. A counter also makes the exploration
   order deterministic across runs.
2. Made BFS use the **same `SearchResult` dataclass** so Task 5 is a strict
   apples-to-apples comparison.
3. Added `frontier_pushes` so Task 5 can distinguish "A* expanded fewer
   states" from "A* also inserted fewer into the frontier".

---

## Task 3 — Test results

Captured verbatim in [`outputs/test_results.log`](outputs/test_results.log).

### Test 1 — Original warehouse

```
################
#S****#........#
#.###*#.########
#...#*#.......##
###.#*########.#
#...#*********##
#.###########*##
#............G##
################
```

| metric | value |
|---|---|
| solution found | **yes** |
| path length    | 18 |
| path cost      | 18 |
| states expanded | 25 |

### Test 2 — Trivial case

Map `#####\n#SG##\n#####`. A* returns path length 1 after expanding 2 states. ✅

### Test 3 — No solution

`G` is walled off. A* returns **no path**, terminates (does not loop), expanding 9
reachable states. ✅

### Test 4 — Alternative paths

Multi-path corridor map. A* returns a path of length 10; BFS, run independently,
confirms 10 is the optimum. ✅

---

## Task 4 — Where every A* concept lives in the code

| Concept | Where in [`search.py`](search.py) |
|---|---|
| State | `Pos = tuple[int, int]` |
| Action | `MOVES` list `[("Up",(-1,0)), ...]` |
| Transition | `Grid.neighbours` yields successors that pass `Grid.passable` |
| Goal test | `n == g` inside `astar` |
| `g(n)` | `g_cost[nb] = g_cost[n] + 1` |
| `h(n)` | the `h: Callable` passed to `astar`; defaults to `h_manhattan` |
| `f(n)` | `tentative + h(nb, g)` pushed onto the heap |
| Frontier | `heapq` of `(f, counter, state)` |
| Visited states | `closed: set[Pos]`, plus the "better `g`" check in `g_cost` |
| Path reconstruction | `_reconstruct(parent, end)` walks the `parent` dict back to `s₀` |

**(a) Data structure for the frontier:** min-heap (Python `heapq`).
**(b) Next state to expand:** the heap's minimum-`f` state, with the
`counter` tie-breaker ensuring stable, deterministic selection.
**(c) Where the heuristic is calculated:** inside `h_manhattan(nb, g)` when
pushing a successor onto the frontier; evaluated once per push.
**(d) Does the program explicitly compute `f = g + h`?** Yes — the sum is the
first element of each heap entry.
**(e) How is repeated exploration prevented?** Two mechanisms. `closed` skips
states already expanded, and the condition `tentative < g_cost[nb]` skips
successors that cannot improve the best known cost.

---

## Task 5 — BFS vs A*

Captured in [`outputs/experiments.log`](outputs/experiments.log).

| algorithm | found? | path length | path cost | states expanded |
|---|---|---|---|---|
| BFS            | yes | 18 | 18 | **37** |
| A* (Manhattan) | yes | 18 | 18 | **25** |

- **(a) Did both find a solution?** Yes.
- **(b) Same length?** Yes — both are optimal here.
- **(c) Fewer expansions?** A* expanded 25 vs BFS's 37.
- **(d) Why?** BFS treats every unexplored cell equally; it expands outward in
  concentric rings regardless of where the goal is. A* orders the frontier by
  `f = g + h`, so cells whose *estimated* total cost to the goal is high are
  deferred. On this map that lets A* skip the cells inside the empty bay in the
  upper right, which BFS still sweeps through.

A* does not win because it is "smarter" in general — it wins because its
heuristic lets it ignore directions that cannot improve on the current best
total cost. If `h = 0` the ordering carries no information and A* reduces to
Dijkstra, which on a uniform-cost grid matches BFS (confirmed in Task 6).

---

## Task 6 — Heuristic study

| algorithm | found? | path length | path cost | states expanded |
|---|---|---|---|---|
| A* with `h = 0` (≡ Dijkstra) | yes | 18 | 18 | 37 |
| A* with Manhattan             | yes | 18 | 18 | 25 |
| A* with Euclidean             | yes | 18 | 18 | 25 |
| A* with `2 × Manhattan`       | yes | 18 | 18 | 26 |

**Reading the table:**

- **`h = 0`.** A* collapses to Dijkstra. On a uniform-cost grid this behaves
  exactly like BFS, so the expand count matches BFS (37). Still optimal — the
  heuristic is admissible (trivially: `0 ≤ h*(n)` always).
- **Manhattan.** Admissible on a 4-connected grid because every direct move
  costs 1 and changes `|dr|+|dc|` by at most 1, so `|dr|+|dc| ≤ true cost`.
  Dominant heuristic here — expands fewest states.
- **Euclidean.** Also admissible since `sqrt(dr² + dc²) ≤ |dr| + |dc|`, so
  Euclidean is pointwise ≤ Manhattan. Still finds the optimum, but is a
  **weaker lower bound**, so A* explores at least as many states; in this map
  the counts happen to tie, but on more complex maps Euclidean typically loses.
- **`2 × Manhattan`.** **Inadmissible** — `h` can exceed the true remaining
  cost. On this specific map A* still returns a 18-step path, but the optimality
  guarantee is gone: with a different map/start/goal pair you can construct a
  case where this heuristic commits to a corridor early and returns a strictly
  longer path. The expand count also went slightly *up* (26 vs 25) because the
  over-estimate caused A* to re-open a cell when a shorter `g` was found later.

**On admissibility.** `h ≤ h*` is the sufficient condition for A* optimality on
a tree-search; with a closed set you also want **consistency**
(`h(n) ≤ cost(n, n') + h(n')`), which Manhattan satisfies on a 4-connected
grid. Breaking admissibility trades the optimality guarantee for raw speed;
whether that is a good deal depends entirely on the problem.

---

## Task 7 — Evaluating the LLM-generated agent

1. **Correct immediately.** Grid parsing, the move delta table, the dataclass
   design, and the `heapq`-based A* loop. These are all textbook patterns that
   the LLM has seen thousands of times and reproduces reliably.

2. **Bugs / design problems found.** The draft used
   `heapq.heappush(frontier, (f, state))` with no tie-breaker, which only
   worked because `Pos = tuple[int,int]` happens to be totally ordered. If I
   later change state representation (say, to a `frozenset` for a STRIPS
   variant), the heap starts throwing `TypeError: '<' not supported`. Added
   a `counter`. The draft also missed `frontier_pushes` entirely, so Task 5's
   claim "A* expands fewer states" could not be distinguished from "A* just
   pushes less onto the frontier".

3. **How I discovered them.** The counter bug surfaced only once I tried
   swapping heuristics (Task 6), because ties became much more common with
   `h = 0`. The missing push counter surfaced the first time I tried to
   explain *why* A* beat BFS and realised I had only one measurement.

4. **Terminology I did not understand.** None in this lab — I had already
   seen `heapq`, closed sets, and path reconstruction in the lecture.

5. **Modifications.** The two mentioned above, plus making BFS use the same
   `SearchResult` dataclass so Task 5 is apples-to-apples.

6. **Most useful tests.** Test 3 (no solution) and Test 4 (alt paths). The
   first catches a planner that silently loops; the second catches a planner
   that returns *a* path instead of the *shortest* path — a classic failure
   mode for A* implementations that use a plain priority queue without the
   `tentative < g_cost[nb]` relaxation step.

7. **Could I trust it without testing?** No. Nothing in the generated code tells
   me whether the heap entries carry `g` or `f`, or whether `g` is updated when
   a better path to the same state is found. Both are easy to get wrong and
   both silently produce a valid-looking (just non-optimal) answer.

8. **What I understood about A* after implementing it.** That A*'s edge over
   BFS is **not** about being smarter per expansion — it is strictly about
   **ordering the frontier**. The expensive part of BFS on a sparse grid is
   not the per-cell work; it is the number of cells. A* with a tight admissible
   `h` prunes the cells that are provably not on the shortest path, and that's
   the entire story.

---

## Final reflection (Section 6)

1. **Why formulate the problem first?** Because the search algorithm is just
   plumbing; the semantics are in `S, A, T, G, c`. Writing BFS or A* first and
   then "fitting it to the problem" is how you end up with an implementation
   that silently assumes 8-connectivity, uniform cost, or deterministic
   transitions when the real problem does not.

2. **In what sense is A* "informed"?** A* uses an estimate `h(n)` of the cost
   from `n` to the goal. BFS is uninformed — it has no sense of direction, just
   of distance from the start. Any search algorithm that consults a non-trivial
   `h` is informed.

3. **Why does the choice of heuristic matter?** The heuristic controls both
   **correctness** (admissibility ⇒ A* returns an optimal path) and
   **efficiency** (tighter admissible heuristics expand fewer states). An
   admissible heuristic that is close to `h*` is near-optimal in expansions;
   an inadmissible one may be faster but can return a wrong (non-shortest) answer.

4. **What did the LLM contribute?** Boilerplate: dataclasses, parsing, the
   `heapq` scaffolding, the move table. Fast to write, easy to audit. The
   algorithmic choices — closed set vs reopening, tie-breaking, which metrics
   to report — were decisions I made and verified.

5. **What could go wrong if I just accepted LLM code without testing?** Any of:
   - priority queue with no tie-breaker crashes on ties;
   - `g` not updated when a cheaper path to the same state is found → A*
     returns a non-shortest path;
   - `closed` set not consulted → infinite re-expansion on graphs with cycles;
   - heuristic computed with the wrong goal cached from an earlier test → silent
     slowdown or wrong result.

   Every one of these produces **output that looks right**. The point of
   Tests 3 and 4 is that output looking right is not enough.

**The lab's one-line summary — `AI Science → AI Engineering` — is exactly
right: the scientific object is `A* on this problem class`; the engineering
object is this particular Python program; the LLM helps with the engineering,
not with the science.**
