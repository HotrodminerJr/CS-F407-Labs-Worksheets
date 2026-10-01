"""Task 3: three tests for the LLM-generated planner.

A — Solvable warehouse problem (plan exists)
B — Impossible problem (no PickUp -> planner must report 'No plan found')
C — Irrelevant actions (extra Move actions must not fool the planner into
    treating 'robot at C' as 'package at C')
"""
from pathlib import Path

from planner import (
    Action,
    plan_bfs,
    verify_plan,
    pretty_plan,
    pretty_state,
    warehouse_actions,
    WAREHOUSE_INITIAL,
    WAREHOUSE_GOAL,
)

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


def banner(txt: str) -> str:
    bar = "=" * len(txt)
    return f"\n{bar}\n{txt}\n{bar}"


def report(name: str, initial, goal, plan, trace) -> str:
    out = [banner(f"{name}")]
    out.append(f"initial state : {pretty_state(frozenset(initial))}")
    out.append(f"goal          : {pretty_state(frozenset(goal))}")
    if plan is None:
        out.append("plan found    : NO  (planner returned 'No plan found')")
        return "\n".join(out)
    out.append(f"plan found    : YES  ({len(plan)} actions)")
    out.append("\nState-transition trace:")
    out.append(pretty_plan(frozenset(initial), plan))
    ok, msg = verify_plan(initial, plan, goal)
    out.append(f"\nindependent verification: {ok}  ({msg})")
    return "\n".join(out)


# ---------- Test A: solvable ----------
def test_a() -> str:
    actions = warehouse_actions()
    plan, trace = plan_bfs(WAREHOUSE_INITIAL, actions, WAREHOUSE_GOAL)
    return report("Test A — Solvable warehouse problem",
                  WAREHOUSE_INITIAL, WAREHOUSE_GOAL, plan, trace)


# ---------- Test B: impossible (no PickUp) ----------
def test_b() -> str:
    actions = warehouse_actions(with_pickup=False)
    plan, trace = plan_bfs(WAREHOUSE_INITIAL, actions, WAREHOUSE_GOAL)
    return report("Test B — Impossible problem (PickUp removed)",
                  WAREHOUSE_INITIAL, WAREHOUSE_GOAL, plan, trace)


# ---------- Test C: irrelevant actions ----------
def test_c() -> str:
    """Add actions that look like Move but do nothing useful. The planner must
    not confuse 'At(Robot,C)' with 'At(Package,C)'."""
    actions = warehouse_actions()
    # Already present. Just to be explicit, we also run the robot alone (no PickUp)
    # to confirm the planner reports failure when the robot ends at C but the package
    # does not.
    actions_no_handling = warehouse_actions(with_pickup=False, with_drop=False)
    plan_full, _ = plan_bfs(WAREHOUSE_INITIAL, actions, WAREHOUSE_GOAL)
    plan_robot_only, _ = plan_bfs(WAREHOUSE_INITIAL, actions_no_handling, WAREHOUSE_GOAL)

    out = [banner("Test C — Irrelevant actions do not fool the planner")]
    out.append(
        "With the full action set, the planner picks up and drops the package "
        "(a plan exists)."
    )
    out.append(f"  plan length (full actions)       : {len(plan_full)}")
    out.append(
        "With PickUp and Drop removed, the robot can still move to C, "
        "but the package stays at A — so the goal At(Package,C) is NOT met "
        "and the planner must report failure (not substitute At(Robot,C))."
    )
    out.append(f"  plan found (robot-only actions)  : {plan_robot_only is not None}")
    out.append(
        f"  interpretation                   : "
        f"{'FAIL — planner incorrectly accepted robot-only moves' if plan_robot_only is not None else 'PASS — planner correctly reported no plan'}"
    )
    return "\n".join(out)


def main() -> None:
    report_text = "\n".join([test_a(), test_b(), test_c()])
    print(report_text)
    (OUT / "test_results.log").write_text(report_text + "\n")
    print(f"\nsaved outputs/test_results.log")


if __name__ == "__main__":
    main()
