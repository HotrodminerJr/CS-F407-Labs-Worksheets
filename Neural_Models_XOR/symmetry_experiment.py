"""Task 4C: identical (zero) weight initialisation keeps hidden units identical."""
from pathlib import Path
import torch
from xor_baseline import train


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    print("=== Symmetry experiment: all weights initialised to 0 ===\n")
    res = train(hidden_act="sigmoid", seed=0, steps=600, lr=0.1,
                zero_init=True, verbose=False)

    print(f"final loss (zero-init) = {res['final_loss']:.4f}")
    print(f"all four correct?      = {res['all_correct']}")
    print(f"final probs            = {[round(p,4) for p in res['probs']]}\n")

    print("fc1.weight (rows = the two hidden units) across training:")
    for step, W in res["W1_rows_log"]:
        r0, r1 = W[0].tolist(), W[1].tolist()
        identical = torch.allclose(W[0], W[1])
        print(f"  step {step:>3}: row0={[round(v,5) for v in r0]}  "
              f"row1={[round(v,5) for v in r1]}  identical={identical}")

    r0, r1 = res["W1_final"][0], res["W1_final"][1]
    print(f"\nfinal rows identical? {torch.allclose(r0, r1)}")
    print("\nExplanation: with all weights equal, both hidden units compute the same")
    print("pre-activation and the same gradient at every step, so they move in lockstep")
    print("and can never represent different features. Random init breaks the symmetry.")

    # Save a short summary file for the report
    with open(out / "symmetry_log.txt", "w") as f:
        for step, W in res["W1_rows_log"]:
            f.write(f"step {step}\n{W.numpy()}\n\n")
        f.write(f"final rows identical? {torch.allclose(r0, r1)}\n")
        f.write(f"final loss {res['final_loss']:.4f}\n")
    print("\nsaved outputs/symmetry_log.txt")


if __name__ == "__main__":
    main()
