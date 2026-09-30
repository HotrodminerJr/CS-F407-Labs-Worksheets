# Lab: Bayesian Networks and Autoregressive Language Models — Answers

Course: CS F407 (AI Laboratory)
Student: Rajat Upadhyay

The worksheet is included as `BN_lab.pdf`. Runnable code is in
`first_order.py`, `second_order.py`, `tests.py`, `compare.py`. Program
outputs are saved under `outputs/`.

---

## Q1 — Why is the autoregressive decomposition useful for generating text?

Text is produced one token at a time. The chain-rule factorisation
`P(x_1..x_T) = P(x_1) · ∏_{t≥2} P(x_t | x_1..x_{t-1})` turns the intractable
problem of drawing a whole sentence from a joint distribution into a
sequence of small next-token problems. We only ever need a *conditional*
distribution over the vocabulary, and we can generate by sampling one
token, appending it to the context, and repeating. It is also what
enables *streaming* generation: we emit token *t* before ever considering
token *t + 1*.

## Q2 — Independence assumption of the first-order network

The graph `X_1 → X_2 → … → X_T` asserts that each token is conditionally
independent of everything before its immediate predecessor given that
predecessor. Formally,

    P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1})   for all t ≥ 2,

or, in independence notation, `X_t ⟂ X_1..X_{t-2} | X_{t-1}`. This is
the first-order Markov assumption.

## Q3 — First-order conditional distributions

Computed by `first_order.py` from the 6-sentence corpus (see
`outputs/first_order_report.txt`):

| current word | P(next \| current) |
|--------------|--------------------|
| `the`  | cat 0.333, dog 0.250, mat 0.167, rug 0.167, park 0.167 |
| `cat`  | sat 0.667, ran 0.333 |
| `dog`  | sat 0.667, ran 0.333 |
| `sat`  | on 1.000 |
| `ran`  | to 1.000 |

Zero-probability transitions are common — e.g. `P(mat | cat) = 0`,
`P(the | sat) = 0`, `P(cat | ran) = 0`. Any (context, word) pair not
listed above has probability 0 under the maximum-likelihood estimate.
This is the sparsity problem that motivates smoothing.

## Q4 — Where are transition counts stored?

In `first_order.count_transitions`, they live in
`counts: dict[str, dict[str, int]]`, a nested `defaultdict(int)` keyed by
`counts[previous_token][next_token]`.

## Q5 — Where is P(X_t | X_{t-1}) computed?

In `first_order.build_cpt`. For each context `prev`, it divides every
`counts[prev][nxt]` by `sum(counts[prev].values())`, producing the CPT
`cpt: dict[str, dict[str, float]]`.

## Q6 — How does the program choose the next word?

Both. `predict_most_probable` returns `argmax_w P(w | context)` (greedy),
while `sample_next` draws from `P(· | context)` via
`random.choices(tokens, weights)`. `generate(..., mode="greedy")` uses
the first; `mode="sample"` uses the second.

The difference: greedy is deterministic — same context, same token — so
it collapses onto whatever loop maximises local probability (in our
corpus, `the → cat → sat → on → the → cat → …`, forever). Sampling is
stochastic; low-probability continuations still occur, so we get
variety and can even reach the `<END>` token.

## Q7 — What happens for an unseen context?

`sample_next` and `predict_most_probable` return `None`. `generate`
treats that as end-of-sentence and stops. Practically this means an
n-gram model cannot generalise past the exact contexts it saw during
training — the classic zero-count problem, addressed in real systems by
smoothing (Laplace, Kneser–Ney) or by backing off to a lower-order model.

## Q8 — What does a total of 0.87 tell you?

That the implementation is wrong, not that the model is uncertain. A
valid CPT is normalised by construction: `Σ_v P(v | w) = 1` is an
identity, not something to be estimated. A total of 0.87 means the code
dropped mass somewhere — a common cause is dividing by the corpus total
instead of by the row total, or filtering the numerator without
filtering the denominator to match, or applying smoothing to some
entries but not adjusting the denominator.

`tests.py` verifies this invariant for both models; both pass to nine
decimal places.

## Q9 — Are argmax predictions what you would expect?

Sometimes. `argmax P(w | the) = cat` matches intuition; so does
`argmax P(w | sat) = on`. But `argmax P(w | on) = the` is not because
"on" *linguistically* demands "the" — it is because every sentence in
this corpus contains "on the". The model measures *frequency in this
corpus*, not *plausibility to a speaker of English*. A larger and more
varied corpus, or a model with richer context, would narrow (but never
eliminate) that gap: probability models capture regularities in data,
which is neither the same as, nor a proxy for, human judgement.

## Q10 — Greedy vs sampling

Greedy generation always produces the same string
(`the cat sat on the cat sat on …`) — it cycles through the local
argmaxes and never terminates. Sampling produces diverse sentences and
frequently terminates by reaching `<END>` (see `outputs/first_order_report.txt`,
20 sampled sentences). Sampling has variation because it draws from the
whole conditional distribution instead of always taking its mode; that
same property is why sampling can wander into low-probability territory,
which is the tradeoff a real system tunes with temperature or top-p.

## Q11 — Second-order vs first-order

1. **Graph:** the first-order model is a chain `X_{t-1} → X_t`; the
   second-order model has two parents, `X_{t-2} → X_t ← X_{t-1}`,
   giving a v-structure at every position.
2. **CPT:** indexed by ordered pairs `(X_{t-2}, X_{t-1})`, not single
   tokens; so entries look like `P(mat | on, the) = 0.333` instead of
   `P(mat | the) = 0.167`.
3. **Context available:** two tokens instead of one — enough to
   distinguish `on the ___` from `to the ___`.
4. **Data required:** larger. In the worst case the number of contexts
   grows from `|V|` to `|V|²`, so many more sentences are needed before
   most contexts are ever observed. On our 6-sentence corpus this
   already shows up (see comparison below).

## Q12 — More context, harder estimation

More context sharpens the conditional distribution because the model
can use information that a shorter context throws away. The tradeoff is
that the CPT has `|V|^k` rows for context length k; the number of
observations per row falls exponentially, so most rows are unseen or
based on a single count. That is exactly what `compare.py` shows: the
second-order model has 18 nonzero parameters and 3 observed bigram
contexts with *no* continuation at all, and its sentence diversity over
50 samples drops from 0.46 to 0.12 — it has essentially memorised the
six training sentences. The estimation problem is
bias–variance in probabilistic form: bigger context reduces bias, but
raises variance unless the data grows to match.

## Q13 — Why Approach B is preferable

Approach B ("implement P(X_t | X_{t-1}) from transition counts with
sampling-based generation") gives you leverage the LLM cannot supply
on its own:

- **Specifying behaviour** turns a code request into an acceptance test —
  you know what "correct" looks like before you read the output.
- **Understanding the representation** lets you spot the bug when the LLM
  stores counts in the wrong dict shape (I nearly shipped a bigram-count
  table indexed by `(prev, next)` tuples that would have made per-context
  totals impossible to check without regrouping).
- **Validating the generated implementation** is only meaningful if you
  know what to validate: here, that CPT rows sum to 1, that
  `argmax P(w | w')` is deterministic, and that sampling actually reaches
  `<END>`.
- **Testing probabilistic invariants** turns "it looked fine" into a
  reproducible check — `tests.py` runs in a fraction of a second and
  would catch any regression instantly.
- **Distinguishing implementation from model** means the code is
  swappable (dictionaries today, a tensor tomorrow) without changing
  what the *model* is; the LLM cannot draw that boundary for you.

## Q14 — What did the Bayesian-network view add?

Three (of the listed) contributions stand out for this lab:

- **A factorisation of the joint distribution.** The BN told us exactly
  which conditional probabilities to estimate — no more, no less — and
  the chain-rule identity guarantees they compose back into a valid
  joint distribution. Without it we would have no principled way of
  saying what "the model" is.
- **A way to reason about independence assumptions.** Drawing the graph
  makes the first-order assumption *visible*: any inaccuracy the model
  shows can be traced back to `X_t ⟂ X_1..X_{t-2} | X_{t-1}`. Adding a
  second parent is not an ad-hoc change to the code; it is a change in
  the graph, and it forces the corresponding change in the CPT.
- **A way to test whether an implementation matches its specification.**
  The CPT rows summing to 1, the correct set of parents for each node,
  the sampling procedure being consistent with the ancestral ordering —
  each of these is a property of the *graph*, so we can test them
  without ever running the model end to end. That is what
  `tests.py` exploits.

## Reflection on LLM use

I drafted `first_order.py` with the help of Claude Code by handing it
the behavioural spec from Part V rather than a general "write a language
model" request. Two things came out of that:

1. The first LLM draft of `second_order.py` used `padded = [START, *sent[1:]]`
   inside `count_triples`, which is a no-op — it just re-drops and
   re-adds the START token that was already there. The generator seeds
   itself with the context `(START, START)`, so the model never saw its
   own initial context and produced empty sentences for every seed.
   I caught this by running the model rather than trusting it (running
   `second_order.py` printed twenty empty strings), then fixed it to
   `padded = [START, *sent]` so `(START, START) → the` is a legal
   transition. The fix is a one-line change in
   [second_order.py](second_order.py), but without an executable
   invariant to check, it is exactly the kind of bug that a reader would
   miss on inspection.
2. The second, subtler observation is that the LLM's greedy generator
   never terminated. This is not a bug in the LLM — it is a property of
   the *model*: greedy decoding on this corpus enters a cycle
   `the → cat → sat → on → the → …` and no `<END>` transition is ever
   the argmax. Recognising it as a model property rather than a code
   problem is only possible if you kept the probabilistic specification
   and the implementation conceptually separate — which is exactly the
   argument in Q13.
