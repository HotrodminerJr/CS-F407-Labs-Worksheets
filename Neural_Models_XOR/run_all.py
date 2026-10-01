"""Run every experiment end-to-end and tee the console output into outputs/."""
import subprocess
import sys
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

jobs = [
    ("baseline_sigmoid", [sys.executable, "xor_baseline.py",
                          "--hidden", "sigmoid", "--seed", "1", "--steps", "4000", "--lr", "0.5"]),
    ("symmetry",         [sys.executable, "symmetry_experiment.py"]),
    ("activations",      [sys.executable, "activation_experiment.py"]),
    ("three_class",      [sys.executable, "three_class_extension.py"]),
]

for name, cmd in jobs:
    print(f"\n########## {name} ##########")
    log_path = OUT / f"{name}.log"
    with open(log_path, "w") as f:
        p = subprocess.run(cmd, capture_output=True, text=True)
        f.write(p.stdout)
        if p.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(p.stderr)
    sys.stdout.write(open(log_path).read())
    if p.returncode != 0:
        print(f"[{name}] FAILED with returncode {p.returncode}")
        sys.exit(p.returncode)
    print(f"[{name}] saved -> {log_path}")
