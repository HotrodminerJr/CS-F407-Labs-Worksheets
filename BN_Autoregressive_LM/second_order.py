"""Second-order autoregressive language model.

Estimates P(X_t | X_{t-2}, X_{t-1}) from triple counts on the toy corpus.
No ML libraries are used.

Corresponding BN factorisation for a sequence of four tokens:
  P(X1..X4) = P(X1) P(X2|X1) P(X3|X1,X2) P(X4|X2,X3)
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from first_order import START, END, load_sentences


def count_triples(sentences: list[list[str]]
                  ) -> dict[tuple[str, str], dict[str, int]]:
    """C((w_{t-2}, w_{t-1}), w_t)."""
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for sent in sentences:
        # Prepend an extra START so that (START, START) predicts the first
        # real word — this matches how generate() seeds its context.
        padded = [START, *sent]
        for a, b, c in zip(padded, padded[1:], padded[2:]):
            counts[(a, b)][c] += 1
    return counts


def build_cpt(counts: dict[tuple[str, str], dict[str, int]]
              ) -> dict[tuple[str, str], dict[str, float]]:
    cpt: dict[tuple[str, str], dict[str, float]] = {}
    for context, next_counts in counts.items():
        total = sum(next_counts.values())
        cpt[context] = {nxt: c / total for nxt, c in next_counts.items()}
    return cpt


def predict_most_probable(cpt, context):
    if context not in cpt:
        return None
    return max(cpt[context].items(), key=lambda kv: kv[1])[0]


def sample_next(cpt, context, rng):
    if context not in cpt:
        return None
    tokens, weights = zip(*cpt[context].items())
    return rng.choices(tokens, weights=weights, k=1)[0]


def generate(cpt, mode="sample", seed=None, max_len=20):
    rng = random.Random(seed)
    out: list[str] = []
    a, b = START, START
    for _ in range(max_len):
        if mode == "greedy":
            nxt = predict_most_probable(cpt, (a, b))
        else:
            nxt = sample_next(cpt, (a, b), rng)
        if nxt is None or nxt == END:
            break
        out.append(nxt)
        a, b = b, nxt
    return out


def main() -> None:
    here = Path(__file__).parent
    sentences = load_sentences(here / "dataset.txt")
    counts = count_triples(sentences)
    cpt = build_cpt(counts)

    out_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        out_lines.append(line)

    emit("=" * 60)
    emit("Second-order autoregressive language model")
    emit("=" * 60)
    emit(f"training sentences: {len(sentences)}")
    emit(f"distinct observed 2-token contexts: {len(cpt)}")

    emit("\n--- Conditional distributions for a few contexts ---")
    for ctx in [(START, "the"), ("the", "cat"), ("the", "dog"),
                ("cat", "sat"), ("dog", "ran"), ("on", "the")]:
        emit(f"  P(next | {ctx}):")
        if ctx not in cpt:
            emit("    (unseen context)")
            continue
        for nxt, p in sorted(cpt[ctx].items(), key=lambda kv: -kv[1]):
            emit(f"    {nxt:<10} {p:.4f}")

    emit("\n--- Normalisation test ---")
    for ctx, dist in sorted(cpt.items()):
        emit(f"  sum P(v | {ctx}) = {sum(dist.values()):.6f}")

    emit("\n--- Sampled generations ---")
    for i in range(20):
        sent = generate(cpt, mode="sample", seed=2000 + i)
        emit(f"  {i+1:2d}: {' '.join(sent)}")

    emit("\n--- Greedy generations ---")
    for i in range(5):
        sent = generate(cpt, mode="greedy", seed=None)
        emit(f"  {i+1}: {' '.join(sent)}")

    (here / "outputs" / "second_order_report.txt").write_text(
        "\n".join(out_lines) + "\n"
    )


if __name__ == "__main__":
    main()
