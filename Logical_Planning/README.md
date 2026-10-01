# Logical Reasoning for Planning

Lab for CS F407 covering STRIPS-style planning, BFS state-space search, and
independent plan verification (with an optional Prolog cross-check).

## The idea in one line

> **Logic decides what is possible; search decides what to try.**

## Files

| File | Role |
|---|---|
| [`logic_lab_ex.pdf`](logic_lab_ex.pdf) | Original worksheet |
| [`answers.md`](answers.md) | Full write-up, prompts, reflection |
| [`planner.py`](planner.py) | STRIPS planner: `applicable`, `apply`, `plan_bfs`, `verify_plan` |
| [`run_tests.py`](run_tests.py) | Tests A/B/C from the worksheet |
| [`planner.pl`](planner.pl) | Prolog verifier for the optional Section 7 |
| [`outputs/test_results.log`](outputs/test_results.log) | Captured output of `run_tests.py` |

## Reproduce — Python

```bash
python run_tests.py
```

No non-stdlib dependencies.

## Reproduce — Prolog (optional)

```bash
sudo apt install swi-prolog
swipl planner.pl
?- can_move(a, b).            % true
?- can_move(a, c).            % false
?- verify_plan([move(a,b), move(b,c)]).   % true
```

## Headline result

The BFS planner solves the warehouse problem in **4 actions** —
`PickUp → Move(A,B) → Move(B,C) → Drop` — matching the hand-constructed
lower bound. With `PickUp` removed, it correctly reports **No plan found**
rather than inventing an action. The irrelevant-actions test confirms the
planner does **not** confuse `At(Robot, C)` with `At(Package, C)`, which is
the key failure mode an LLM-written planner is most likely to introduce.
