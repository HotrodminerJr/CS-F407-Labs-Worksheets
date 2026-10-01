"""Task 2-4A/B: train a 2-2-1 MLP on XOR, print loss, probabilities and one gradient tensor."""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])


class XORNet(nn.Module):
    def __init__(self, hidden_act="sigmoid"):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.hidden_act = {
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "relu": torch.relu,
        }[hidden_act]

    def forward(self, x):
        h = self.hidden_act(self.fc1(x))
        return self.fc2(h)


def train(hidden_act="sigmoid", seed=0, steps=4000, lr=0.1, zero_init=False, verbose=True):
    torch.manual_seed(seed)
    net = XORNet(hidden_act=hidden_act)
    if zero_init:
        for p in net.parameters():
            nn.init.zeros_(p)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr)

    losses = []
    early_grad_norm = None
    W1_rows_log = []

    for step in range(steps):
        opt.zero_grad()
        logits = net(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        if step == 5:
            early_grad_norm = net.fc1.weight.grad.detach().norm().item()
        if zero_init and step in (0, 1, 2, 5, 50, 500):
            W1_rows_log.append((step, net.fc1.weight.detach().clone()))
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        final_logits = net(X)
        probs = torch.sigmoid(final_logits).squeeze()
        preds = (probs > 0.5).int()

    if verbose:
        print(f"=== hidden={hidden_act}  seed={seed}  zero_init={zero_init} ===")
        print(f"initial loss = {losses[0]:.4f}")
        print(f"final   loss = {losses[-1]:.4f}")
        print("inputs  | prob  | pred | target")
        for xi, p, pr, yi in zip(X.tolist(), probs.tolist(), preds.tolist(), Y.squeeze().tolist()):
            print(f"{xi}  | {p:.4f} | {pr}    | {int(yi)}")
        print(f"all four correct? {(preds.float() == Y.squeeze()).all().item()}")
        print(f"early-step ||grad W1||_2 = {early_grad_norm:.6f}")

    return {
        "losses": losses,
        "probs": probs.tolist(),
        "preds": preds.tolist(),
        "final_loss": losses[-1],
        "all_correct": bool((preds.float() == Y.squeeze()).all().item()),
        "early_grad_norm": early_grad_norm,
        "W1_final": net.fc1.weight.detach().clone(),
        "W1_grad": net.fc1.weight.grad.detach().clone(),
        "W1_rows_log": W1_rows_log,
        "net": net,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", default="sigmoid", choices=["sigmoid", "tanh", "relu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    res = train(hidden_act=args.hidden, seed=args.seed, steps=args.steps, lr=args.lr)

    # Backprop check: print one gradient tensor and interpret it.
    print("\n--- Backpropagation check (Part B) ---")
    print("fc1.weight.grad = dL/dW^(1), shape:", tuple(res["W1_grad"].shape))
    print(res["W1_grad"].numpy())
    print("Because BCEWithLogitsLoss uses the mean over the 4 examples,")
    print("this grad is (1/4) * sum_i dL_i/dW^(1).")

    # Save loss curve
    plt.figure(figsize=(6, 3.5))
    plt.plot(res["losses"])
    plt.xlabel("step")
    plt.ylabel("BCE loss")
    plt.title(f"XOR loss curve  (hidden={args.hidden}, seed={args.seed})")
    plt.tight_layout()
    plt.savefig(out / f"loss_{args.hidden}_seed{args.seed}.png", dpi=120)
    print(f"\nsaved {out}/loss_{args.hidden}_seed{args.seed}.png")


if __name__ == "__main__":
    main()
