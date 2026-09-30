"""First-order autoregressive language model.

Implements P(X_t | X_{t-1}) from transition counts on a small toy corpus.
No ML libraries are used — only dictionaries and random sampling.

Deliverables covered:
  - Parts IV-IX of the worksheet
  - CPT construction, next-word prediction, sampling-based generation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

START = "<START>"
END = "<END>"


def load_sentences(path: str | Path) -> list[list[str]]:
    """Read the corpus, lowercase it, and wrap each sentence in START/END."""
    sentences: list[list[str]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip().lower()
        if not line:
            continue
        sentences.append([START, *line.split(), END])
    return sentences


def count_transitions(sentences: list[list[str]]) -> dict[str, dict[str, int]]:
    """C(w_i, w_j) = number of times w_j follows w_i."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sent in sentences:
        for prev, nxt in zip(sent, sent[1:]):
            counts[prev][nxt] += 1
    return counts


def build_cpt(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)."""
    cpt: dict[str, dict[str, float]] = {}
    for prev, next_counts in counts.items():
        total = sum(next_counts.values())
        cpt[prev] = {nxt: c / total for nxt, c in next_counts.items()}
    return cpt


def show_distribution(cpt: dict[str, dict[str, float]], word: str) -> None:
    """Pretty-print P(X_t | X_{t-1}=word)."""
    if word not in cpt:
        print(f"  (no observed transitions from {word!r})")
        return
    dist = sorted(cpt[word].items(), key=lambda kv: -kv[1])
    print(f"  P(next | {word!r}):")
    for nxt, p in dist:
        print(f"    {nxt:<10} {p:.4f}")


def predict_most_probable(cpt: dict[str, dict[str, float]], word: str) -> str | None:
    """argmax_w P(w | word)."""
    if word not in cpt:
        return None
    return max(cpt[word].items(), key=lambda kv: kv[1])[0]


def sample_next(cpt: dict[str, dict[str, float]], word: str,
                rng: random.Random) -> str | None:
    """Sample the next token from the CPT."""
    if word not in cpt:
        return None
    tokens, weights = zip(*cpt[word].items())
    return rng.choices(tokens, weights=weights, k=1)[0]


def generate(cpt: dict[str, dict[str, float]], mode: str = "sample",
             seed: int | None = None, max_len: int = 20) -> list[str]:
    """Generate a sentence starting from START.

    mode = "sample": sample from P(w | previous).
    mode = "greedy": always take argmax P(w | previous).
    """
    rng = random.Random(seed)
    out: list[str] = []
    prev = START
    for _ in range(max_len):
        if mode == "greedy":
            nxt = predict_most_probable(cpt, prev)
        elif mode == "sample":
            nxt = sample_next(cpt, prev, rng)
        else:
            raise ValueError(f"unknown mode: {mode}")
        if nxt is None or nxt == END:
            break
        out.append(nxt)
        prev = nxt
    return out


def main() -> None:
    here = Path(__file__).parent
    sentences = load_sentences(here / "dataset.txt")
    counts = count_transitions(sentences)
    cpt = build_cpt(counts)

    out_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        out_lines.append(line)

    emit("=" * 60)
    emit("First-order autoregressive language model")
    emit("=" * 60)
    emit(f"training sentences: {len(sentences)}")
    vocab = sorted({tok for sent in sentences for tok in sent})
    emit(f"vocabulary ({len(vocab)}): {vocab}")

    emit("\n--- Q3: Conditional probability distributions ---")
    for w in ["the", "cat", "dog", "sat", "ran"]:
        show_distribution(cpt, w)
        line = f"  P(next | {w!r}) -> "
        if w in cpt:
            line += ", ".join(f"{k}={v:.3f}" for k, v in
                              sorted(cpt[w].items(), key=lambda kv: -kv[1]))
        else:
            line += "(unseen context)"
        out_lines.append(line)

    emit("\n--- Zero-probability transitions (for the words above) ---")
    for w in ["the", "cat", "dog", "sat", "ran"]:
        if w not in cpt:
            emit(f"  {w!r}: never appears as a context")
            continue
        seen = set(cpt[w])
        missing = [v for v in vocab if v not in seen and v not in (START,)]
        emit(f"  {w!r}: {len(missing)} zero-prob next tokens (e.g. "
             f"{missing[:6]}{'...' if len(missing) > 6 else ''})")

    emit("\n--- Q8-style normalisation test ---")
    for word, dist in sorted(cpt.items()):
        total = sum(dist.values())
        emit(f"  sum P(v | {word!r}) = {total:.6f}")

    emit("\n--- Most-probable next word for five contexts ---")
    for w in ["the", "cat", "dog", "sat", "on"]:
        pred = predict_most_probable(cpt, w)
        emit(f"  argmax P(w | {w!r}) = {pred!r}")

    emit("\n--- Sampled generations (Mode B) ---")
    for i in range(20):
        sent = generate(cpt, mode="sample", seed=1000 + i)
        emit(f"  {i+1:2d}: {' '.join(sent)}")

    emit("\n--- Greedy generations (Mode A) ---")
    for i in range(5):
        sent = generate(cpt, mode="greedy", seed=None)
        emit(f"  {i+1}: {' '.join(sent)}")

    (here / "outputs" / "first_order_report.txt").write_text(
        "\n".join(out_lines) + "\n"
    )


if __name__ == "__main__":
    main()
