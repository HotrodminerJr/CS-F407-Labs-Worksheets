# LLM-Assisted Code Generation for Bayesian Networks

Lab for CS F407. The scientific object is a tiny Sprinkler Bayesian network;
the engineering question is: *can we trust an LLM to write the `pgmpy` code,
and how do we verify what it produced?*

## Files

| File | Role |
|---|---|
| [`llm_bn.ipynb`](llm_bn.ipynb) | Original lab notebook (reference model, LLM code-gen, validation, deliberately-broken demos) |
| [`reference_bn.py`](reference_bn.py) | Trusted Sprinkler BN + VE inference + enumeration cross-check + MLE sample-size study + broken-model check — runs **offline** |
| [`outputs/reference_run.log`](outputs/reference_run.log) | Captured output |
| [`answers.md`](answers.md) | Full write-up: division of labour, validation gates, failure modes, reflection |

## Reproduce the reference (no LLM, no GPU)

```bash
pip install "pgmpy>=1.0" numpy pandas matplotlib
python reference_bn.py
```

## Headline results

- `pgmpy` variable elimination and independent 16-way enumeration of the joint
  agree on `P(R=1 | W=1) = 0.7048` to machine epsilon (`|diff| = 2.22e-16`).
- Explaining-away works as expected:
  `P(R=1 | W=1) = 0.70` → `P(R=1 | W=1, S=0) = 0.99`.
- MLE for `P(R=1 | C=1)` (true value 0.8) tightens with sample size:
  `0.826 → 0.755 → 0.773 → 0.804` for `N = 50, 200, 1000, 5000`.
- A deliberately broken CPT whose column sums to 1.2 is correctly rejected
  with `ValueError: Sum ... is not equal to 1`.

## In one line

> *Program runs ≠ probabilistic model is correct ≠ scientific assumptions are appropriate.*

The LLM can accelerate the implementation step. It does not remove the need
for an independently constructed oracle to validate the result.
