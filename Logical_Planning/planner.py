"""STRIPS-style planner with BFS search.

A planning problem (I, A, G) is solved by a textbook recipe:
  * state  = frozenset of ground logical propositions
  * action = (name, pos_pre, neg_pre, pos_eff, neg_eff)
  * action a is applicable in state S iff pos_pre(a) ⊆ S and neg_pre(a) ∩ S = ∅
  * apply(S, a) = (S − neg_eff(a)) ∪ pos_eff(a)
  * BFS over states, remembering parent + action to reconstruct the plan

Design notes:
  - A proposition is a string like "At(Robot, A)". Strings are hashable, so
    `frozenset[str]` gives us free O(1) membership and immutable states that
    can live in a `set` used as the visited frontier.
  - Negative preconditions let us express "the robot is NOT already holding
    the package", which the PickUp action needs to be safe.
  - BFS guarantees the shortest plan (fewest actions).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, List, Optional, Tuple

State = FrozenSet[str]


@dataclass(frozen=True)
class Action:
    name: str
    pos_pre: FrozenSet[str] = field(default_factory=frozenset)
    neg_pre: FrozenSet[str] = field(default_factory=frozenset)
    pos_eff: FrozenSet[str] = field(default_factory=frozenset)
    neg_eff: FrozenSet[str] = field(default_factory=frozenset)

    def applicable(self, s: State) -> bool:
        # Logical component: S ⊨ Preconditions(a)
        return self.pos_pre <= s and self.neg_pre.isdisjoint(s)

    def apply(self, s: State) -> State:
        # S' = Apply(S, a) = (S − neg_eff) ∪ pos_eff
        return frozenset((s - self.neg_eff) | self.pos_eff)


def goal_satisfied(state: State, goal: FrozenSet[str]) -> bool:
    return goal <= state


def plan_bfs(
    initial: Iterable[str],
    actions: List[Action],
    goal: Iterable[str],
    max_states: int = 100_000,
) -> Tuple[Optional[List[Action]], List[State]]:
    """Return (plan, trace) or (None, []) if no plan exists within max_states."""
    I: State = frozenset(initial)
    G: FrozenSet[str] = frozenset(goal)

    if goal_satisfied(I, G):
        return [], [I]

    # queue of (state, path_of_actions)
    frontier: deque[Tuple[State, List[Action]]] = deque([(I, [])])
    visited = {I}

    while frontier:
        if len(visited) > max_states:
            return None, []
        s, path = frontier.popleft()

        for a in actions:
            if not a.applicable(s):
                continue
            s2 = a.apply(s)
            if s2 in visited:
                continue
            new_path = path + [a]
            if goal_satisfied(s2, G):
                trace = replay(I, new_path)
                return new_path, trace
            visited.add(s2)
            frontier.append((s2, new_path))

    return None, []


def replay(initial: State, plan: List[Action]) -> List[State]:
    """Return the sequence of states visited by running `plan` from `initial`."""
    trace = [initial]
    s = initial
    for a in plan:
        assert a.applicable(s), f"replay error: {a.name} not applicable in {sorted(s)}"
        s = a.apply(s)
        trace.append(s)
    return trace


def verify_plan(
    initial: Iterable[str],
    plan: List[Action],
    goal: Iterable[str],
) -> Tuple[bool, str]:
    """Independent verification: step through the plan and confirm each precondition.

    This is deliberately a *separate* function from `plan_bfs` so it can be used
    to double-check plans that came from any other source (another planner, a
    hand-written plan, an LLM's suggestion).
    """
    s: State = frozenset(initial)
    for i, a in enumerate(plan):
        if not a.applicable(s):
            missing_pos = a.pos_pre - s
            spurious_neg = a.neg_pre & s
            return False, (
                f"step {i} ({a.name}) not applicable in state {sorted(s)}; "
                f"missing positive preconditions: {sorted(missing_pos)}, "
                f"forbidden propositions present: {sorted(spurious_neg)}"
            )
        s = a.apply(s)
    if not goal_satisfied(s, frozenset(goal)):
        return False, f"final state {sorted(s)} does not entail goal {sorted(goal)}"
    return True, "plan verified: every precondition met and goal reached"


# ---------- warehouse problem ----------
LOCATIONS = ["A", "B", "C"]
EDGES = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]  # bidirectional, no A<->C


def move(frm: str, to: str) -> Action:
    return Action(
        name=f"Move({frm},{to})",
        pos_pre=frozenset({f"At(Robot,{frm})"}),
        pos_eff=frozenset({f"At(Robot,{to})"}),
        neg_eff=frozenset({f"At(Robot,{frm})"}),
    )


def pickup(loc: str) -> Action:
    return Action(
        name=f"PickUp(Package,{loc})",
        pos_pre=frozenset({f"At(Robot,{loc})", f"At(Package,{loc})"}),
        neg_pre=frozenset({"Holding(Package)"}),  # don't pick up what you already hold
        pos_eff=frozenset({"Holding(Package)"}),
        neg_eff=frozenset({f"At(Package,{loc})"}),
    )


def drop(loc: str) -> Action:
    return Action(
        name=f"Drop(Package,{loc})",
        pos_pre=frozenset({f"At(Robot,{loc})", "Holding(Package)"}),
        pos_eff=frozenset({f"At(Package,{loc})"}),
        neg_eff=frozenset({"Holding(Package)"}),
    )


def warehouse_actions(with_pickup: bool = True, with_drop: bool = True) -> List[Action]:
    acts: List[Action] = [move(a, b) for a, b in EDGES]
    if with_pickup:
        acts += [pickup(loc) for loc in LOCATIONS]
    if with_drop:
        acts += [drop(loc) for loc in LOCATIONS]
    return acts


WAREHOUSE_INITIAL = {"At(Robot,A)", "At(Package,A)"}
WAREHOUSE_GOAL = {"At(Package,C)"}


def pretty_state(s: State) -> str:
    return "{" + ", ".join(sorted(s)) + "}"


def pretty_plan(initial: State, plan: List[Action]) -> str:
    lines = [f"S0 = {pretty_state(initial)}"]
    s = initial
    for i, a in enumerate(plan, 1):
        s = a.apply(s)
        lines.append(f"a{i} = {a.name}")
        lines.append(f"S{i} = {pretty_state(s)}")
    return "\n".join(lines)
