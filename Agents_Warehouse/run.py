"""Task 3: run the generated program on the warehouse and record what happens."""
from pathlib import Path

from agent import GoalBasedAgent, bfs, parse, render

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    grid = parse()
    print("Warehouse map (as given in the lab):\n")
    print(render(grid))

    # Pure planning (what the goal-based agent would compute)
    actions, trace, expanded = bfs(grid)
    assert actions is not None and trace is not None

    print(f"\nSearch algorithm        : BFS over 4-connected grid cells")
    print(f"States expanded         : {expanded}")
    print(f"Plan length (actions)   : {len(actions)}")
    print(f"Start                   : {grid.start}")
    print(f"Goal                    : {grid.goal}")

    print("\nAction sequence:")
    print("  " + " -> ".join(actions))

    print("\nFinal map with path overlaid (* = cells visited):\n")
    print(render(grid, trace))

    # Execute the plan with the agent to confirm it actually works.
    agent = GoalBasedAgent(grid)
    pos = grid.start
    executed: list[tuple[str, tuple[int, int]]] = []
    while True:
        a = agent.choose_action()
        if a is None:
            break
        deltas = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
        dr, dc = deltas[a]
        pos = (pos[0] + dr, pos[1] + dc)
        assert grid.passable(pos), f"illegal move into wall at {pos}"
        agent.perceive(pos)
        executed.append((a, pos))

    print(f"\nAgent executed {len(executed)} actions and reached {pos}.")
    print(f"Reached goal? {pos == grid.goal}")

    log = [
        "Warehouse map:",
        render(grid),
        "",
        f"Search algorithm        : BFS over 4-connected grid cells",
        f"States expanded         : {expanded}",
        f"Plan length (actions)   : {len(actions)}",
        f"Start                   : {grid.start}",
        f"Goal                    : {grid.goal}",
        "",
        "Action sequence:",
        "  " + " -> ".join(actions),
        "",
        "Final map with path overlaid (* = cells visited):",
        render(grid, trace),
        "",
        f"Agent executed {len(executed)} actions and reached {pos}.",
        f"Reached goal? {pos == grid.goal}",
    ]
    (OUT / "run.log").write_text("\n".join(log) + "\n")
    print("\nsaved outputs/run.log")


if __name__ == "__main__":
    main()
