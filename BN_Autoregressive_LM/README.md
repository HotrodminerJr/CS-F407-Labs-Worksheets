# Bayesian Networks and Autoregressive Language Models

AI Laboratory worksheet: first- and second-order n-gram language models
built from probability first principles, no ML libraries.

## Files

- [BN_lab.pdf](BN_lab.pdf) — the original worksheet.
- [dataset.txt](dataset.txt) — the 6-sentence training corpus.
- [first_order.py](first_order.py) — first-order model `P(X_t | X_{t-1})`,
  CPT construction, greedy and sampling generation.
- [second_order.py](second_order.py) — second-order model
  `P(X_t | X_{t-2}, X_{t-1})`.
- [tests.py](tests.py) — probability-normalisation tests
  (`Σ_v P(v | context) = 1`) for both models.
- [compare.py](compare.py) — parameter counts, unseen contexts, and
  sentence-diversity comparison.
- [answers.md](answers.md) — written answers to Questions 1–14 and a
  reflection on LLM usage.
- [outputs/](outputs/) — reports emitted by the scripts above.

## Run

```bash
python3 first_order.py
python3 second_order.py
python3 tests.py
python3 compare.py
```

No dependencies beyond the Python standard library (Python 3.9+).
