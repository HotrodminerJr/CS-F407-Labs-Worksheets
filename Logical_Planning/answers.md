# Lab — Logical Reasoning for Planning

**Course:** CS F407 — Artificial Intelligence
**Worksheet:** [`logic_lab_ex.pdf`](logic_lab_ex.pdf)

**Code in this folder:**

| File | Role |
|---|---|
| [`planner.py`](planner.py) | STRIPS-style planner with BFS, state `Apply`, independent `verify_plan` |
| [`run_tests.py`](run_tests.py) | Tests A, B, C from the worksheet |
| [`planner.pl`](planner.pl) | Prolog verifier (optional Section 7, Tasks 6–8) |
| [`outputs/test_results.log`](outputs/test_results.log) | Captured output of `run_tests.py` |

**Reproduce (Python):**

```bash
python run_tests.py
```

**Reproduce (Prolog, optional):**

```bash
sudo apt install swi-prolog   # or your package manager's equivalent
swipl planner.pl
?- can_move(a,b).
?- can_move(a,c).
?- verify_plan([move(a,b), move(b,c)]).
```

---

## Task 0 — Understand the planning problem

- **Initial state** `I = { At(Robot, A), At(Package, A) }`.
- **Goal** `G = { At(Package, C) }`.
- **Actions** `A`:

  | Action | Positive preconditions | Negative preconditions | Positive effects | Negative effects |
  |---|---|---|---|---|
  | `Move(X, Y)` for each `connected(X, Y)` | `At(Robot, X)` | – | `At(Robot, Y)` | `At(Robot, X)` |
  | `PickUp(Package, L)` | `At(Robot, L), At(Package, L)` | `Holding(Package)` | `Holding(Package)` | `At(Package, L)` |
  | `Drop(Package, L)` | `At(Robot, L), Holding(Package)` | – | `At(Package, L)` | `Holding(Package)` |

  The map is `A ⇄ B ⇄ C` (no direct `A ⇄ C` edge). `Move` is instantiated once per
  edge; `PickUp` and `Drop` once per location.

- **Negative precondition on PickUp.** The lab description doesn't require this, but
  without it `PickUp(Package, A)` is still "applicable" the moment after you've
  already picked the package up — it would strip `At(Package, A)` from an already-
  stripped state and loop. Adding `¬Holding(Package)` keeps the state space finite
  and keeps the plan short.

### Which actions are initially applicable?

In `I = { At(Robot, A), At(Package, A) }`:

- `PickUp(Package, A)` — **applicable**. Both positive preconditions are in `I`
  and `Holding(Package)` is not.
- `Drop(Package, C)` — **not applicable**. The positive preconditions
  `At(Robot, C)` and `Holding(Package)` are both missing from `I`.
- `Move(A, B)` — applicable (`At(Robot, A) ∈ I`).

---

## Task 1 — Plan constructed by hand

| Step | Action | State after action |
|---|---|---|
| S₀ | — | `{ At(Robot, A), At(Package, A) }` |
| S₁ | `PickUp(Package, A)` | `{ At(Robot, A), Holding(Package) }` |
| S₂ | `Move(A, B)` | `{ At(Robot, B), Holding(Package) }` |
| S₃ | `Move(B, C)` | `{ At(Robot, C), Holding(Package) }` |
| S₄ | `Drop(Package, C)` | `{ At(Robot, C), At(Package, C) }` |

`S₄ ⊨ G` because `At(Package, C) ∈ S₄`. ✅

Any shorter plan is impossible: the goal requires `At(Package, C)`, so the
package must be moved from `A` to `C`, which requires `PickUp` + at least two
`Move`s (A→B, B→C) + `Drop`. 4 actions is the lower bound, and BFS confirms it.

---

## Task 2 — Prompt used for the LLM

> *I want to implement a simple STRIPS-style planning agent in Python. Represent
> a state as a frozenset of logical propositions (ground strings). Each action
> should be a frozen dataclass containing a name, positive preconditions,
> negative preconditions, positive effects, and negative effects. An action is
> applicable in state S iff pos_pre ⊆ S and neg_pre ∩ S = ∅. Applying an action
> computes `(S − neg_eff) ∪ pos_eff`. Use breadth-first search over states to
> find a shortest sequence of actions achieving a goal (another set of
> propositions). The program should (a) detect when no plan exists, (b) print
> the resulting action sequence, (c) print the state reached after each action,
> and (d) provide a separate verify_plan function that re-runs an arbitrary
> plan step-by-step and reports which precondition failed if one does. Set a
> `max_states` safety limit on the frontier. Keep the code library-free.*

**Where each specification idea lives in the generated code:**

| Spec idea | Code location |
|---|---|
| **Precondition check** — "when is an action applicable?" | `Action.applicable` in [`planner.py`](planner.py) |
| **Effect application** — "how does the state change?" | `Action.apply` |
| **Goal test** — "when does planning terminate?" | `goal_satisfied` + the `goal_satisfied(s2, G)` branch inside `plan_bfs` |
| **BFS** — "how are alternatives explored?" | the `deque` / `visited` loop in `plan_bfs` |

**Modifications made after reading the draft, before accepting it:**

1. Added a **negative precondition** `¬Holding(Package)` on `PickUp`. Without it
   the planner re-explores pointlessly after it already picks the package up.
2. Added a dedicated `verify_plan` function that is **independent of the search
   loop** so the same checker can validate a plan produced by hand, by the LLM
   itself, or by another planner. This is the "generate ↔ verify" split the lab
   is really about.
3. Made `State = FrozenSet[str]` so visited sets stay hashable and the whole
   search is immutable + replayable.

---

## Task 3 — Test results

Captured verbatim in [`outputs/test_results.log`](outputs/test_results.log).
Summary:

### Test A — Solvable warehouse problem

Found plan of length 4, matching the hand-built plan exactly:

```
PickUp(Package,A) → Move(A,B) → Move(B,C) → Drop(Package,C)
```

Independent `verify_plan` confirms every precondition is satisfied in the
state in which the action is executed, and the final state entails
`{ At(Package, C) }`. ✅

### Test B — Impossible problem (PickUp removed)

With `PickUp` removed, the robot can still roam `A ⇄ B ⇄ C`, but the package
is stuck at `A`. The planner correctly reports **No plan found** rather than
inventing a `PickUp`-like action or substituting `At(Robot, C)` for the goal. ✅

### Test C — Irrelevant actions

The crucial check: when `PickUp`/`Drop` are both removed, the planner should
*not* accept a trajectory that merely gets the robot to `C` as if it were a
plan. Observed:

- Full action set → plan of length 4 (correct).
- Robot-only action set → `plan found: False` (correct).

The planner does not confuse `At(Robot, C)` with `At(Package, C)`. ✅

---

## Task 4 — Where logic and search each enter

```
Current state
       ↓
Check action preconditions          ←  LOGICAL REASONING: S ⊨ Pre(a)
       ↓
Compute successor state  S' = Apply(S, a)   ←  LOGICAL REASONING: effect application
       ↓
Generate successor state
       ↓
Search over alternatives            ←  SEARCH: BFS enumerates applicable actions
       ↓
Goal?                               ←  LOGICAL REASONING: S_n ⊨ G
```

The missing step in the lab's diagram is **"Compute successor state
`S' = Apply(S, a)`"** — removing negative effects and adding positive ones is
itself a logical operation on the state's proposition set.

**In one line:** *Logic decides what is possible; search decides what to try.*

Search without logic can't tell applicable from inapplicable actions, so it
either considers every action in every state (explosion) or silently relaxes
the model (unsafe plans). Logic without search can validate a single candidate
plan but cannot *find* one — there are factorially many candidate orderings.

---

## Task 5 — "Can the LLM verify its own plan?"

I asked the LLM to generate a plan and separately asked it to produce an
*English explanation* of why its plan was valid, action by action. For the
easy warehouse case the explanation agreed with my executed trace, so this
check was uninformative. The real test is **Test C** above: when `PickUp` is
missing, the LLM is strongly inclined to patch the explanation
("the robot picks up the package when it arrives at `C`") because
narratively that's what *should* happen — but the formal state transitions
executed by `planner.py` show no such action exists.

**Which should you trust more — the LLM's explanation or the executed
state transitions?** The executed state transitions. The LLM is a generator
with no obligation to actually run the preconditions, so it can produce
confident, plausible, even internally consistent prose that happens to be
false. The Python code's state transitions were produced by a program with
no sense of "what sounds reasonable" — only `pos_pre ⊆ S` and
`neg_pre ∩ S = ∅`. That is why the lab's parting line is

> **A generated explanation is not the same as an independent verification.**

---

## Reflection questions (Section 5)

1. **Why specify preconditions and effects *before* calling the LLM?**
   The spec is what makes the generated code testable. Without one you have
   no way to tell whether the LLM wrote a correct planner or a convincing-looking
   wrong one, because every output is a sequence of plausible Python.

2. **Example of a bug caused by skipping precondition checks.** In Test A, if
   `Drop` is applied without checking `Holding(Package)`, the planner can
   "drop" the package at `C` while the package is still physically at `A`,
   producing a state like `{ At(Robot, A), At(Package, C) }`. The plan passes
   the goal check but is physically impossible.

3. **Why "looks reasonable" ≠ "valid".** Validity is a property of state
   transitions, not of narrative. A plan is valid only if every intermediate
   state entails the next action's preconditions. Humans (and LLMs) can be
   convinced by a plan whose preconditions were never actually checked.

4. **What the LLM contributed.** Boilerplate for the `Action` dataclass, the
   BFS skeleton, and the state pretty-printer. These are all substitutable with
   any STRIPS tutorial — fast to write, little room for subtle error, and
   cheap to review line by line.

5. **What I had to verify independently.** Every semantic claim:
   - that `applicable` and `apply` really implement `S ⊨ Pre(a)` and
     `(S − neg_eff) ∪ pos_eff`;
   - that **Test B** actually returns "no plan" and does not time out silently;
   - that **Test C** does not confuse `At(Robot, C)` with `At(Package, C)` —
     this is exactly the kind of bug an LLM is likely to introduce by
     "simplifying" the goal check;
   - that `verify_plan` is independent of the search loop so it would catch a
     corrupt plan from any source.

6. **Where logical reasoning is used in this lab.**
   - `Action.applicable`: `S ⊨ Pre(a)` — entailment check.
   - `Action.apply`: `S' = (S − neg_eff) ∪ pos_eff` — logical state update.
   - `goal_satisfied`: `S ⊨ G` — entailment check again.
   - Prolog verifier (Task 7): `verify_plan([move(a,b), move(b,c)])` is a
     literal entailment query against the warehouse facts.

7. **Relation to the search module.** The planner is BFS over the induced
   state graph, with `expand(S) = { Apply(S, a) : S ⊨ Pre(a) }`. Replace BFS
   with depth-first and you get a different planner with the same soundness
   (every plan it returns is valid) but worse completeness guarantees on
   infinite action spaces. Replace it with A* and a heuristic (e.g. Manhattan
   distance between package and goal) and you get an informed planner. The
   *logic* in the planner is independent of which search algorithm sits on top.

---

## Section 7 — Prolog verifier (optional extension)

I did not have `swi-prolog` installed on this machine, so I could not execute
`planner.pl` to produce a log. The queries that the file is designed to answer:

| Query | Expected answer | Why |
|---|---|---|
| `?- can_move(a, b).` | `true` | `connected(a, b)` is a fact. |
| `?- can_move(a, c).` | `false` | No `connected(a, c)` fact, and the rule only consults `connected/2`. |
| `?- valid_move(a, b).` | `true` | Same reasoning. |
| `?- valid_move(b, c).` | `true` | Same reasoning. |
| `?- valid_move(a, c).` | `false` | Same reasoning. |
| `?- verify_plan([move(a,b), move(b,c)]).` | `true` | Both steps are supported by `connected/2`. |
| `?- verify_plan([move(a,c)]).` | `false` | `move(a,c)` isn't supported — the Python planner would be *wrong* to propose it. |
| `?- reduce_speed.` | `true` | `wet_road` is a fact; `wet_road ⇒ slippery` and `slippery ⇒ reduce_speed`. |

### Section 7 reflection

1. **Fact vs. rule.** A fact is an unconditional proposition (`wet_road.`).
   A rule says "the head holds whenever the body holds"
   (`slippery :- wet_road.` ≡ `wet_road → slippery`).

2. **Queries and entailment.** Asking `?- animal(polly).` is asking the Prolog
   engine whether `animal(polly)` is entailed by the facts and rules in the
   knowledge base. Prolog answers `true` if it can derive it, `false` (strictly:
   *not provable from this KB*) otherwise.

3. **Why use Prolog to verify a Python plan?** Because the Python program *generated*
   the plan — using the same program to also certify it is circular. Running
   the plan through Prolog's independent `connected`/`valid_move` rules is a
   *different* implementation of the same semantics; if both agree, the plan is
   much more likely to be correct. If they disagree, exactly one of them is
   wrong, and now you know to look.

4. **Why an independent verifier matters when the plan came from an LLM.** The
   LLM generates with no obligation to check preconditions. A separate, rule-based
   verifier has no incentive to be plausible — it only answers `true` or `false`
   against explicit facts and rules. That is the "`Generate → Independent
   verification`" architecture: the generator can be fast and heuristic; the
   verifier is slow and sound.

### Task 8 — Prolog inference chain

Query `?- reduce_speed.` succeeds via:

```
wet_road                     (fact)
wet_road  ⇒  slippery        (rule: slippery :- wet_road.)
slippery  ⇒  reduce_speed    (rule: reduce_speed :- slippery.)
∴ reduce_speed               (conclusion)
```

This is exactly the schema `Fact ⇒ Rule ⇒ Rule ⇒ Conclusion.` the worksheet asks for.
