"""Task 4D: compare sigmoid / tanh / ReLU hidden activations on XOR."""
from pathlib import Path
from xor_baseline import train


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    SEED = 1
    STEPS = 4000
    LR = 0.5
    for act in ("sigmoid", "tanh", "relu"):
        res = train(hidden_act=act, seed=SEED, steps=STEPS, lr=LR, verbose=False)
        rows.append((act, res["final_loss"], res["all_correct"], res["early_grad_norm"]))

    header = f"{'activation':<10} {'final loss':>12} {'4/4?':>6} {'||gradW1||_2 (step 5)':>22}"
    print(header)
    print("-" * len(header))
    for act, fl, ok, g in rows:
        print(f"{act:<10} {fl:>12.4f} {str(ok):>6} {g:>22.6f}")

    with open(out / "activation_table.md", "w") as f:
        f.write(f"Seed={SEED}, steps={STEPS}, lr={LR}, optimiser=SGD, loss=BCEWithLogits\n\n")
        f.write("| hidden activation | final loss | 4/4 correct? | early ||grad W1||_2 |\n")
        f.write("|---|---|---|---|\n")
        for act, fl, ok, g in rows:
            f.write(f"| {act} | {fl:.4f} | {ok} | {g:.6f} |\n")
    print("\nsaved outputs/activation_table.md")

    print("\nInterpretation (for this tiny XOR run — do NOT generalise):")
    print("  sigmoid: derivatives peak at 0.25 and shrink fast when units saturate,")
    print("           so early gradients are small and convergence is slow.")
    print("  tanh:    zero-centred, derivative up to 1, usually trains faster than sigmoid.")
    print("  ReLU:    derivative is exactly 1 on the active side, 0 on the inactive side;")
    print("           larger early gradient, but if a unit starts inactive it can 'die'.")


if __name__ == "__main__":
    main()
