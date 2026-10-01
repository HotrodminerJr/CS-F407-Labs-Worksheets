# Lab — LLM-Assisted Code Generation for Bayesian Networks

**Course:** CS F407 — Artificial Intelligence
**Notebook (original):** [`llm_bn.ipynb`](llm_bn.ipynb)
**Reproducible reference (no LLM required):** [`reference_bn.py`](reference_bn.py)

| File | Role |
|---|---|
| [`llm_bn.ipynb`](llm_bn.ipynb) | Full lab notebook (reference model, LLM code-gen, validation, broken-model demos) |
| [`reference_bn.py`](reference_bn.py) | Trusted Sprinkler BN + inference + estimation + broken-model check; runs offline |
| [`outputs/reference_run.log`](outputs/reference_run.log) | Captured output of the reference script |
| [`answers.md`](answers.md) | This write-up |
| [`README.md`](README.md) | Quick-start |

**Reproduce the reference (no GPU, no LLM download):**

```bash
pip install "pgmpy>=1.0" numpy pandas matplotlib
python reference_bn.py
```

**To also run the LLM cells** (notebook only, needs ~3 GB disk for the small Qwen):

```bash
pip install "transformers>=4.45" accelerate
jupyter notebook llm_bn.ipynb
```

---

## 1 The scientific object — the Sprinkler Bayesian network

```
        Cloudy
       /      \
      v        v
    Rain    Sprinkler
       \      /
        v    v
        WetGrass
```

Joint distribution (chain rule, respecting the DAG):

> `P(C, R, S, W) = P(C) · P(R | C) · P(S | C) · P(W | R, S)`

Numerical parameters (verbatim from the lab):

| CPT | values |
|---|---|
| `P(C=1)` | 0.5 |
| `P(R=1 \| C=0)` / `P(R=1 \| C=1)` | 0.2 / 0.8 |
| `P(S=1 \| C=0)` / `P(S=1 \| C=1)` | 0.5 / 0.1 |
| `P(W=1 \| R, S)` | 0.01, 0.90, 0.90, 0.99 for `(R=0,S=0), (R=0,S=1), (R=1,S=0), (R=1,S=1)` |

---

## 2 Inference — library result vs independent enumeration

The lab's canonical query is `P(R=1 | W=1)`. `pgmpy` computes it with
variable elimination; `reference_bn.py` *also* computes it by summing all `2⁴ = 16`
joint assignments where `W=1`.

| method | `P(R=1 | W=1)` |
|---|---|
| `pgmpy.VariableElimination` | **0.704769** |
| Independent enumeration     | **0.704769** |
| `|diff|` | `2.22 × 10⁻¹⁶` (machine epsilon) |

**Why this matters.** A test oracle you built yourself, independently of the
library, is the only way to tell "the library ran" from "the library
represented the probability distribution you specified". The two agree here to
machine precision, so the reference model is trustworthy — later it is the
yardstick against which any LLM-generated code must agree.

### The four posterior queries from the HW exercise

Computed by the reference model via variable elimination:

| query | value | sanity check |
|---|---|---|
| `P(R=1 | W=1)`         | 0.7048 | wet grass raises belief in rain |
| `P(S=1 | W=1)`         | 0.4278 | wet grass also raises belief in sprinkler |
| `P(C=1 | W=1)`         | 0.5746 | cloudiness is weakly implied via its children |
| `P(R=1 | W=1, S=0)`    | 0.9922 | **explaining away**: with the sprinkler off, rain must explain the wet grass |

The last one is the classic *explaining away* phenomenon in a V-structure:
observing one cause makes an alternative cause less needed. The posterior for
Rain jumps from 0.70 to 0.99 the moment we condition on `S = 0`.

---

## 3 LLM-assisted code generation — division of labour

| Task | Component responsible |
|---|---|
| Specify the probability model (nodes, edges, CPT numbers) | **The scientist (us)** |
| Interpret the natural-language programming request          | **LLM** |
| Generate candidate Python code                              | **LLM** |
| Represent the Bayesian network                              | `pgmpy` |
| Check *local* CPD consistency (columns sum to 1)            | `pgmpy.check_model()` |
| Perform exact inference                                     | `pgmpy.VariableElimination` |
| Check that the implemented model matches the specification  | **The scientist (us)** |

The LLM is a **programmer with no probabilistic responsibility**. It can
produce the Python; it cannot certify that the Python represents the right
distribution. The notebook enforces this by requiring explicit human approval
(`APPROVE_GENERATED_CODE = False` by default) before any generated code is `exec`-ed.

### The exact prompt (from the notebook, cell 21)

> *Write Python code using the current pgmpy API. Construct this binary
> discrete Bayesian network: edges Cloudy → Rain, Cloudy → Sprinkler, Rain →
> WetGrass, Sprinkler → WetGrass. Use states 0=False and 1=True. [full
> numerical table of CPTs]. Then run variable elimination to compute
> `P(Rain | WetGrass=1)`. Assign the model to `generated_model` and the
> result to `generated_posterior`.*

### Why this prompt format

- Repeats the structure *and* the numerical CPTs explicitly, so the LLM has
  no ambiguity about what to construct.
- Names `generated_model` and `generated_posterior` so the validation cell
  can `assert` on them.
- Says *"the current pgmpy API"* so the LLM does not reach for the old
  `BayesianModel` (deprecated) or `inference.VariableElimination` import style.

---

## 4 Validation gates — what the notebook actually checks

| Gate | What it catches |
|---|---|
| Static AST scan (`basic_generated_code_check`) | Rejects `eval`, `exec`, `open`, `__import__`, imports from unexpected packages |
| `generated_model.check_model()` | CPDs locally normalised (columns sum to 1) |
| Structure assertions | nodes = `{Cloudy, Rain, Sprinkler, WetGrass}`, edges exactly as specified |
| CPT shape assertions | Each `TabularCPD` has the expected `variable_card` and `evidence_card` |
| **Semantic test** | `generated_posterior.values[1] ≈ reference_posterior.values[1]` (within `1e-6`) |

The semantic test is the one that cannot be skipped. Everything else can pass
on code that still implements the wrong distribution — e.g. two correctly
normalised CPDs whose rows are swapped. The notebook demonstrates this
explicitly: in cell 58 it builds a `wrong_semantics_model` whose WetGrass CPD
has the four parent-state columns *permuted*. Every column still sums to 1 so
`check_model()` accepts it, but `P(R=1 | W=1)` differs from the reference —
and only the semantic test flags the problem.

---

## 5 Parameter estimation — MLE and the sample-size study

With the graph fixed and the CPT numbers unknown, maximum-likelihood estimates
of binary CPT entries reduce to count ratios:

> `P̂(R=1 | C=1) = N(R=1, C=1) / N(C=1)`

Running MLE on simulated data of growing size (reference script output):

| N       | `P̂(R=1 | C=1)` | error |
|---------|------------------|-------|
| 50      | 0.8261           | +0.026 |
| 200     | 0.7553           | −0.045 |
| 1 000   | 0.7725           | −0.027 |
| 5 000   | **0.8040**       | +0.004 |

The estimate wanders at small `N` and settles toward the true 0.8 as `N` grows
— textbook consistency.

**Sampling variability at a fixed `N = 100`, five random seeds:**

| seed | estimate |
|---|---|
| 1 | 0.6939 |
| 2 | 0.7955 |
| 3 | 0.7447 |
| 4 | 0.7679 |
| 5 | 0.7959 |

Range `[0.69, 0.80]` from the *same* true parameter. The correct explanation
is **sampling variability**, not "the model is wrong" and not "the LLM did
something" — this stays true even when no LLM is involved. The lab deliberately
asks you to practise distinguishing these.

### MLE vs Bayesian (BDeu) at small `N`

With `N = 30` and a rare parent-state configuration, MLE can produce `P = 0`
or `P = 1` from a single empty cell. A Bayesian estimator with Dirichlet
pseudo-counts (`BDeu`, `equivalent_sample_size = 10`) regularises these to
sensible probabilities. The lab asks the LLM to *revise* its MLE code into
Bayesian code — the kind of task where LLMs are strongest, because the change
is local and the spec is precise.

---

## 6 The two failure modes the lab deliberately exhibits

### Deliberately broken: columns do not sum to 1

```python
cpd_rain_broken = TabularCPD(
    variable="Rain", variable_card=2,
    values=[[0.9, 0.2], [0.3, 0.8]],   # column 0 sums to 1.2
    evidence=["Cloudy"], evidence_card=[2],
)
```

The reference script's output:

> `broken model correctly rejected: ValueError: Sum or integral of conditional probabilities for node Rain is not equal to 1.`

`pgmpy.check_model()` catches this one — a locally invalid CPD.

### Subtler: valid numbers, wrong semantics

```python
wrong_wetgrass = TabularCPD(
    variable="WetGrass", variable_card=2,
    values=[
        [0.99, 0.10, 0.01, 0.10],   # columns permuted vs the spec
        [0.01, 0.90, 0.99, 0.90],
    ],
    evidence=["Rain", "Sprinkler"], evidence_card=[2, 2],
)
```

Every column sums to 1 so `check_model()` is happy. But the posterior
`P(R | W=1)` is different from the reference. The only gate that catches this
is the semantic test against the oracle.

### The central lesson (notebook cell 59)

```
Program runs
    !=
Probabilistic model is correct
    !=
Scientific assumptions are appropriate
```

An LLM can make the first implication look trivial while the second and third
quietly fail. The validation stack has to cover all three.

---

## 7 Reflection

### What the LLM contributed

- Boilerplate: `TabularCPD` construction, `.add_cpds(...)`, `check_model()`,
  the `VariableElimination` call, the switch from MLE to `BayesianEstimator`.
  Fast, easy to review, hard to get wrong at this scale.

### What a human had to verify

- **Parent ordering in the CPT.** `pgmpy` interprets columns in lexicographic
  order of `evidence`, so `evidence=["Rain","Sprinkler"]` gives columns
  `(R=0,S=0), (R=0,S=1), (R=1,S=0), (R=1,S=1)`. Flip the evidence list to
  `["Sprinkler","Rain"]` and the columns mean something different — but
  `check_model()` cannot tell.
- **Prior equivalent sample size** on `BayesianEstimator`. A default
  `equivalent_sample_size = 10` is a *modelling choice*, not an implementation
  detail. The LLM may silently pick one; the human has to accept or override it.
- **State coding.** `0 = False` is a convention. If the LLM swaps it, every
  posterior interpretation flips, but the numbers still look reasonable.

### Could the LLM be trusted to also *verify* its own code?

No, for the same reason that an LLM can produce syntactically valid wrong
code: it can also produce confidently wrong *explanations* of that code. The
only reliable verifier is an independent oracle — in this lab, the
hand-constructed enumeration on 16 assignments and the fixed-seed MLE study.

### Where this generalises

Replace "Bayesian network" with any other formally specified system
(a planner, a search algorithm, a database schema, an optimiser). The pattern
is always:

> **specify** (human) → **generate** (LLM) → **inspect** (human + static checker)
> → **execute** (library) → **validate** (semantic test against an oracle)

The LLM compresses the implementation step. It does not compress the
specification step and it does not remove the validation step.
