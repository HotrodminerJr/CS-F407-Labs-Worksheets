# Agents — Goal-Based Warehouse Navigator

Short lab for CS F407: design, prompt, generate, and test a *goal-based* agent
that navigates a warehouse grid from `S` to `G`.

## Files

| File | Role |
|---|---|
| [`agents_lab.pdf`](agents_lab.pdf) | Original worksheet |
| [`answers.md`](answers.md) | Full write-up — spec, block diagram, prompt, results |
| [`agent.py`](agent.py) | `Grid`, BFS, `GoalBasedAgent` (stdlib only) |
| [`run.py`](run.py) | Runs the agent on the lab map, writes `outputs/run.log` |
| [`outputs/run.log`](outputs/run.log) | Captured output |

## Reproduce

```bash
python run.py
```

## Headline result

Found a 20-action plan around two shelving blocks; agent executes the plan and
reaches the goal cell `(1, 19)` from `(1, 1)`. 52 states expanded by BFS.

```
#####################
#S***.#************G#
#.##****##########..#
#.....##............#
#.######.###.#.###..#
#........#..........#
#####################
```

For the deeper A* study on the same shape of problem, see the companion lab in
[`../Search_AStar/`](../Search_AStar/).
