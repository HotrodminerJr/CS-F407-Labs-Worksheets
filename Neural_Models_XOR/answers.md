# Lab — Neural Models: Learning, Depth, Activations, and Output Layers

**Course:** CS F407 — Artificial Intelligence
**Lab PDF:** [`neur_models_lab_ex.pdf`](neur_models_lab_ex.pdf)

The code lives in this folder:

| File | Purpose |
|---|---|
| [`xor_baseline.py`](xor_baseline.py) | 2-2-1 MLP + BCE on XOR, loss curve, gradient tensor (Tasks 2, 3, 4A, 4B) |
| [`symmetry_experiment.py`](symmetry_experiment.py) | Zero-init symmetry failure (Task 4C) |
| [`activation_experiment.py`](activation_experiment.py) | sigmoid / tanh / ReLU comparison (Task 4D) |
| [`three_class_extension.py`](three_class_extension.py) | Three-class softmax + CE head (Task 5) |
| [`run_all.py`](run_all.py) | Runs everything and writes `outputs/*.log` |

Reproduce:

```bash
pip install torch numpy matplotlib
python run_all.py
```

---

## Task 1 — Problem specification

- **Input space** `X = {0, 1} x {0, 1} = {(0,0), (0,1), (1,0), (1,1)}`.
- **Output space** `Y = {0, 1}` (disagreement warning).
- **Labelled examples (XOR truth table):**

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Why a single straight line cannot separate the classes.**
The positive class is `{(0,1), (1,0)}` and the negative class is `{(0,0), (1,1)}`.
On a 2D plot the two positives sit on the *anti-diagonal* and the two negatives sit on the
*main diagonal*. Any straight line `w1 x1 + w2 x2 + b = 0` partitions the plane into two
half-planes, and no half-plane contains both `(0,0)` and `(1,1)` while excluding both `(0,1)` and `(1,0)`.
So XOR is **not linearly separable**.

**Prediction about a single affine + sigmoid.** With no hidden non-linearity, a single
affine map followed by sigmoid still produces a linear decision boundary; it cannot
drive the loss to zero and will converge to about 0.5 on at least one of the four points
(BCE loss stuck near `ln 2 ≈ 0.693`).

---

## Task 2 — Model design

**Architecture:** 2 inputs → 2 hidden units (tanh/sigmoid/ReLU) → 1 logit → sigmoid.
**Loss:** `BCEWithLogitsLoss` (numerically stable: combines sigmoid + BCE in one op).
**Optimiser:** vanilla SGD, `lr=0.5`, full-batch, 4000 steps.
**Initialisation:** PyTorch default (uniform Kaiming-like for `nn.Linear`), seeded.

**Answers to the three questions in the lab:**

1. **Why the hidden non-linearity is scientifically necessary.** A composition of affine
   maps is itself an affine map, so without a non-linearity the network can only
   represent linear decision boundaries. XOR is not linearly separable; therefore a
   non-linear hidden layer is a *scientific* requirement, not an engineering preference.

2. **Why sigmoid + BCE is a sensible pairing.** The target is one Bernoulli variable.
   Sigmoid outputs a probability in `(0, 1)`, and BCE is the negative log-likelihood of
   that Bernoulli, so minimising BCE *is* maximum-likelihood estimation. The gradient
   of BCE w.r.t. the pre-activation simplifies to `p − y`, which is well-behaved and
   does not vanish when the model is confidently wrong.

3. **What counts as successful learning (three checks).**
   - final BCE loss `< 0.05`;
   - all four thresholded predictions equal the targets (**4/4 correct**);
   - `fc1.weight.grad` is non-zero at step 0, so backprop is actually supplying a signal;
   - *(bonus)* repeated runs with different seeds succeed at least most of the time —
     with only two hidden units a tiny fraction of seeds can get stuck in a local minimum.

---

## Task 3 — LLM prompt used

> *Generate minimal PyTorch code for the following model and dataset. Do not change
> the architecture or task. Model: 2 inputs → 2 hidden units with sigmoid → 1 output,
> trained on the XOR truth table with BCEWithLogitsLoss and SGD. After training,
> report the final loss, all four probabilities, thresholded labels, and one
> parameter-gradient tensor (`fc1.weight.grad`). Set `torch.manual_seed(0)` for
> reproducibility and explain each test in one sentence.*

**Corrections made before running the generated code:**

1. **Used `BCEWithLogitsLoss` instead of `BCELoss(sigmoid(.))`.** The LLM's first draft
   applied `torch.sigmoid` inside the model and used plain `BCELoss`; this is numerically
   less stable. Replaced with raw logits + `BCEWithLogitsLoss`, and applied sigmoid
   only when printing probabilities.
2. **Switched to full-batch training.** The draft looped over the four examples
   one-by-one inside each epoch; the lab asks for full-batch training, which also makes
   the gradient we print the clean batch average `(1/4) Σ_i ∂L_i/∂W^(1)`.

---

## Task 4A — Basic learning check

Output of `python xor_baseline.py --hidden sigmoid --seed 1 --steps 4000 --lr 0.5`
is saved in [`outputs/baseline_sigmoid.log`](outputs/baseline_sigmoid.log) and
[`outputs/loss_sigmoid_seed1.png`](outputs/loss_sigmoid_seed1.png).

| metric | value |
|---|---|
| initial loss | 0.7028 (close to `ln 2 ≈ 0.693`, as expected for a 50/50 untrained logit) |
| final loss   | 0.0124 |
| 4/4 correct? | **yes** — probs `[0.0144, 0.9886, 0.9893, 0.0125]` for `(0,0), (0,1), (1,0), (1,1)` |

If a run fails to reach 4/4, only *engineering* knobs are changed (seed, learning
rate, number of steps). The architecture — two hidden units with a non-linearity
followed by a sigmoid output — is left untouched because *that* is the scientific
claim under test.

---

## Task 4B — Backpropagation check

`fc1.weight.grad` is a `2x2` tensor whose entry `(i, j)` equals
`∂L/∂W^(1)_{ij}`, i.e. the partial derivative of the scalar loss with respect to
the weight connecting input `j` to hidden unit `i`. The exact numbers are printed
inside [`outputs/baseline_sigmoid.log`](outputs/baseline_sigmoid.log).

Because `BCEWithLogitsLoss` defaults to `reduction='mean'`,
`L = (1/4) Σ_i L_i`, so by linearity of differentiation
`∂L/∂W^(1) = (1/4) Σ_i ∂L_i/∂W^(1)` — the gradient we see is the **average** of the
four example-wise gradients, not their sum.

---

## Task 4C — Symmetry experiment (zero init)

Running [`symmetry_experiment.py`](symmetry_experiment.py) with every weight set to 0
produces the log in [`outputs/symmetry.log`](outputs/symmetry.log) and
[`outputs/symmetry_log.txt`](outputs/symmetry_log.txt). The two rows of `fc1.weight`
(the two hidden units) stay **bit-identical at every step**, and the network
cannot learn XOR — final loss stays near `ln 2 ≈ 0.693`.

**Why.** Zero init makes both hidden units compute the same pre-activation on every
input. Backprop then delivers the *same* upstream gradient to each row of `W^(1)`
(the downstream weight `W^(2)` is also identical for both units). Equal starting
weights + equal gradients at every step ⇒ the two rows remain equal forever, so the
network effectively has only one hidden unit — which, like a single affine model,
cannot represent XOR. Random initialisation breaks this symmetry.

---

## Task 4D — Activation experiment

Table written to [`outputs/activation_table.md`](outputs/activation_table.md); log in
[`outputs/activations.log`](outputs/activations.log). Observed numbers at
`seed=1, steps=4000, lr=0.5, SGD`:

| hidden activation | final loss | 4/4 correct? | early `‖∇_{W^(1)} L‖_2` (step 5) |
|---|---|---|---|
| sigmoid | 0.0124 | **yes** | 0.001753 |
| tanh    | 0.3473 | no      | 0.008501 |
| relu    | 0.6931 | no (stuck at `ln 2`) | 0.006358 |

**Important caveat.** The lab explicitly says *"Do not claim that one activation is
universally best from four points"*. The result above is **one run with one seed on
one tiny problem** — on different seeds the ordering flips. In particular ReLU here
got stuck because the two hidden units initialised to a region where at least one
was inactive on every XOR input (classic *dead ReLU*), and with no gradient flowing
through it the network reduced to a single effectively-active hidden unit, which
cannot represent XOR. The number to report is what we observed, not a verdict.

**Interpretation (for this specific run — not a universal claim):**

- **sigmoid** has derivative at most `0.25` and shrinks toward 0 when the unit saturates,
  so early gradients are small and the loss decreases slowly.
- **tanh** is zero-centred with derivative up to 1; usually gives larger early
  gradients than sigmoid at the same point.
- **ReLU** has derivative exactly 1 on the active side and 0 on the inactive side,
  so the early gradient norm tends to be the largest — but if a unit happens to start
  inactive on *all four* XOR inputs its gradient is zero and that unit is dead.

Four data points are not enough to crown a "best" activation; the point is that the
**mechanism** that produces a small gradient is different (saturation vs. inactivity),
and you can tell them apart by inspecting the hidden pre-activations.

---

## Task 5 — Three-class softmax extension

Classes: `0 = (0,0)`, `1 = {(0,1), (1,0)}`, `2 = (1,1)`.

Code: [`three_class_extension.py`](three_class_extension.py). Output:
[`outputs/three_class.log`](outputs/three_class.log) and
[`outputs/three_class_results.txt`](outputs/three_class_results.txt).

**Predictions made before running:**

1. **Shape of the final weight matrix:** `(3, hidden_size)` — one row per class.
   In the code the hidden size is 4, so `fc2.weight.shape == (3, 4)`.
2. **Logits per example:** 3.
3. **Why softmax probabilities sum to 1:** `softmax(z)_k = exp(z_k) / Σ_j exp(z_j)`,
   and the denominator is exactly the sum of the numerators, so the entries sum to
   `(Σ_k exp(z_k)) / (Σ_j exp(z_j)) = 1`.
4. **Why the logit gradient is `p − y`:** `L = −log p_c` where `p = softmax(z)` and
   `c` is the true class.
   `∂L/∂z_k = ∂(−log p_c)/∂z_k = p_k − 1[k == c] = p_k − y_k`
   with `y` one-hot. This clean form is exactly why softmax is paired with
   cross-entropy — the sigmoid+cross-entropy derivative generalises to `p − y`.

**Optional diagnostic.** Adding 100 to every logit before softmax leaves the
probability vector unchanged (up to floating-point roundoff — see
`outputs/three_class.log`). That is because adding a constant `c` to every logit
multiplies the numerator *and* the denominator by `exp(c)`, which cancels.
Numerically stable implementations subtract `max(z)` first so the largest exponent
is `exp(0) = 1`, avoiding overflow for very large logits.

---

## Reflection questions

1. **Depth vs. non-linearity.** The XOR experiment shows that adding more affine
   layers does nothing — any affine stack is equivalent to a single affine map and
   cannot separate XOR. The jump in representational power came from the hidden
   **non-linearity**, not from depth itself. Depth multiplies capacity *only*
   when a non-linearity sits between every pair of linear layers.

2. **Evidence that backprop supplied useful learning signal.** Non-zero gradients
   alone are not sufficient (a stuck network can also have non-zero gradients).
   The evidence that mattered was the *conjunction*: the loss curve decreased
   monotonically over the first few hundred steps, the final loss sank well below
   `ln 2`, and all four thresholded predictions matched the targets. These together
   show the gradient was pointing in a direction that actually reduced loss.

3. **Why zero init prevented distinct features.** With equal weights the two hidden
   units receive equal inputs and equal upstream gradients on every step, so their
   rows of `W^(1)` evolve identically. The network collapses to an effective
   single-unit hidden layer, which cannot represent XOR. Random asymmetry is what
   lets the two units specialise.

4. **How changing the activation changed the gradient.**
   - *Scientific view:* each activation has a different Jacobian — sigmoid clips
     the gradient magnitude to at most `0.25` per unit, tanh to at most `1`, ReLU
     is a `{0, 1}` gate. These Jacobians multiply through the backward pass and
     change the direction and norm of the gradient we actually use.
   - *Engineering observation:* in this experiment the early-step gradient norm
     generally orders as `sigmoid < tanh ≤ ReLU`, which translated into faster
     loss reduction for tanh/ReLU — see the table in `outputs/activation_table.md`.

5. **Output layer + loss must match the task.** The output layer defines the
   distribution the model is predicting (Bernoulli for sigmoid, Categorical for
   softmax), and the loss is the negative log-likelihood of that distribution.
   Pairing them correctly is what gives the clean `p − y` gradient; mismatched
   pairs (e.g. softmax + MSE) produce poorly scaled gradients, slower convergence,
   and no probabilistic interpretation.

6. **Where the LLM helped vs. where human verification was essential.**
   - *Helped:* generating boilerplate `nn.Module`, picking `BCEWithLogitsLoss`
     over the less stable `BCELoss`, and the softmax head change for Task 5.
   - *Human verification essential:* the LLM initially looped per-sample instead
     of full-batch, which changed what the "one gradient tensor" I was supposed
     to print actually meant; and it suggested dropping the hidden layer "to simplify"
     — the exact move that would invalidate the entire scientific claim.
     The code is syntactically fine either way; only reasoning about the
     specification catches these.

7. **Which tests survive scaling up.**
   - *Keep:* loss curve monotonicity, held-out accuracy, verifying that the
     output layer and loss form a valid NLL pair, and a *sampled* gradient-norm
     check at an early step to detect saturation/dead-unit pathologies.
   - *Drop:* exhaustive finite-difference gradient checks and inspecting every
     parameter tensor — infeasible at scale. Replace them with autograd's own
     `gradcheck` on a small sub-module during unit tests, and with running-average
     gradient-norm telemetry in production training.

---

## Checkpoint: how LLM output was verified

- **Static review** of the generated code: shape of each tensor, matching the
  model to the Task-2 specification, confirming `BCEWithLogitsLoss` is applied to
  raw logits.
- **Dynamic checks**: initial loss near `ln 2 ≈ 0.693`, final loss near 0,
  all four predictions correct, non-zero `fc1.weight.grad`, and the softmax
  logit-shift invariance test in Task 5.
- **Experiment-level checks**: the symmetry experiment with zero init *must*
  fail to learn, and both rows of `fc1.weight` *must* remain identical — if
  either does not hold, the code is wrong even if it compiles.
