# Neural Models — XOR, Depth, Activations, Output Layers

Lab for CS F407 covering:

- why XOR needs a non-linear hidden layer (depth alone is not enough);
- backpropagation in PyTorch and what `.grad` tensors mean;
- the symmetry pathology caused by identical weight initialisation;
- sigmoid vs. tanh vs. ReLU on the same tiny problem;
- extending the binary head to a 3-class softmax + cross-entropy head.

## Files

| File | Role |
|---|---|
| [`neur_models_lab_ex.pdf`](neur_models_lab_ex.pdf) | Original lab worksheet |
| [`answers.md`](answers.md) | Full write-up (specs, prompts, results, reflections) |
| [`xor_baseline.py`](xor_baseline.py) | Baseline 2-2-1 MLP + BCE (Tasks 2, 3, 4A, 4B) |
| [`symmetry_experiment.py`](symmetry_experiment.py) | Zero-init symmetry failure (Task 4C) |
| [`activation_experiment.py`](activation_experiment.py) | sigmoid / tanh / ReLU table (Task 4D) |
| [`three_class_extension.py`](three_class_extension.py) | 3-class softmax + CE head (Task 5) |
| [`run_all.py`](run_all.py) | Runs every experiment, writes `outputs/*.log` |
| [`outputs/`](outputs/) | Logs, result tables, loss-curve plot |

## Reproduce

```bash
pip install torch numpy matplotlib
python run_all.py
```

All experiments run on CPU and finish in well under a minute.

## Headline result

A 2-2-1 MLP with a non-linear hidden activation and `BCEWithLogitsLoss` learns the
XOR truth table (4/4 correct, loss ≪ `ln 2`). With all weights initialised to
**zero**, the same architecture is permanently stuck — because the two hidden units
start identical and receive identical gradients forever, there is no mechanism by
which they can specialise. The experiment is the clearest small-scale demonstration
that *representation capacity requires both non-linearity and asymmetry*.
