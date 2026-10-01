"""Task 5: three-class softmax head on the same XOR-shaped inputs.

Classes:
  0 -> both sensors inactive  (0,0)
  1 -> sensors disagree       (0,1), (1,0)
  2 -> both sensors active    (1,1)
"""
from pathlib import Path
import torch
import torch.nn as nn

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([0, 1, 1, 2], dtype=torch.long)


class XORThreeClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)      # hidden slightly wider to make 3 clusters easy
        self.fc2 = nn.Linear(4, 3)      # 3 logits per example

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    net = XORThreeClass()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.05)

    for step in range(2000):
        opt.zero_grad()
        logits = net(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = net(X)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    print(f"final CE loss = {loss.item():.4f}")
    print(f"final fc2.weight shape = {tuple(net.fc2.weight.shape)}  (expected (3, 4))")
    print(f"logits per example     = {logits.shape[1]} (expected 3)\n")

    print("inputs   | class probabilities (softmax)                  | pred | target")
    for xi, pr, pd, yi in zip(X.tolist(), probs.tolist(), preds.tolist(), Y.tolist()):
        print(f"{xi}  | [{pr[0]:.4f}, {pr[1]:.4f}, {pr[2]:.4f}] | {pd}    | {yi}")

    s = probs[0].sum().item()
    print(f"\nsoftmax probs for input (0,0) sum to {s:.6f} (should be ~1.0)")

    # Logit-shift invariance
    shifted = logits + 100.0
    probs_shift = torch.softmax(shifted, dim=1)
    diff = (probs - probs_shift).abs().max().item()
    print(f"max |softmax(logits) - softmax(logits+100)| = {diff:.3e}")
    print("Softmax is invariant to adding a constant to every logit, which is why")
    print("stable implementations subtract max(logit) before exp to avoid overflow.")

    print("\nWhy the logit gradient is p - y:")
    print("  CE loss  L = -log p_c where p = softmax(z)")
    print("  dL/dz_k = p_k - 1[k == c] = p_k - y_k   (y one-hot)")
    print("This simple form is the whole reason softmax is paired with cross-entropy.")

    with open(out / "three_class_results.txt", "w") as f:
        f.write(f"final CE loss {loss.item():.4f}\n")
        f.write(f"fc2.weight shape {tuple(net.fc2.weight.shape)}\n\n")
        for xi, pr, pd, yi in zip(X.tolist(), probs.tolist(), preds.tolist(), Y.tolist()):
            f.write(f"{xi} probs={pr} pred={pd} target={yi}\n")
        f.write(f"\nrow0 sum = {s}\n")
        f.write(f"max |p - softmax(z+100)| = {diff}\n")
    print("\nsaved outputs/three_class_results.txt")


if __name__ == "__main__":
    main()
