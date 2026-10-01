# Search and A*

Lab for CS F407. Implements A* search for a warehouse grid-navigation problem,
compares it against BFS, and studies four heuristics.

## Files

| File | Role |
|---|---|
| [`search_lab_ex.pdf`](search_lab_ex.pdf) | Original worksheet |
| [`answers.md`](answers.md) | Full write-up — spec, prompt, results, reflection |
| [`search.py`](search.py) | Grid parser, BFS, A* with pluggable heuristic, 4 heuristics |
| [`tests.py`](tests.py) | Task 3 — original warehouse / trivial / no-solution / alt-paths |
| [`experiments.py`](experiments.py) | Task 5 BFS-vs-A* and Task 6 heuristic study |
| [`outputs/`](outputs/) | Captured logs and solved-map overlay |

## Reproduce

```bash
python tests.py
python experiments.py
```

Pure stdlib.

## Headline result

| algorithm | path length | states expanded |
|---|---|---|
| BFS                       | 18 | 37 |
| A* (`h = 0`, ≡ Dijkstra)  | 18 | 37 |
| A* (Manhattan)            | 18 | **25** |
| A* (Euclidean)            | 18 | 25 |
| A* (`2 × Manhattan`, inadmissible) | 18 | 26 |

All four A* variants happen to find the same optimum on this specific map, but
only the admissible heuristics (`h = 0`, Manhattan, Euclidean) carry a
guarantee. `2 × Manhattan` can return a non-optimal path on a differently shaped
map — the point of Task 6 is that **working output ≠ validated algorithm**.
