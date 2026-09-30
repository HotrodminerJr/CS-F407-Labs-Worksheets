"""Compare first-order and second-order models (Part XIII)."""

from __future__ import annotations

from pathlib import Path

import first_order
import second_order


def diversity(sentences: list[str]) -> float:
    """Fraction of distinct sentences in a list."""
    return len(set(sentences)) / len(sentences) if sentences else 0.0


def main() -> None:
    here = Path(__file__).parent
    sents = first_order.load_sentences(here / "dataset.txt")

    fo_cpt = first_order.build_cpt(first_order.count_transitions(sents))
    so_cpt = second_order.build_cpt(second_order.count_triples(sents))

    fo_params = sum(len(d) for d in fo_cpt.values())
    so_params = sum(len(d) for d in so_cpt.values())

    vocab = sorted({tok for s in sents for tok in s})
    # A "zero-probability context" for the first-order model is a vocabulary
    # word that never appears as a previous token.
    fo_zero_ctx = [w for w in vocab if w not in fo_cpt and w != "<END>"]
    # For the second-order model, count 2-token contexts drawn from the
    # observed vocabulary that were never seen as (prev-prev, prev).
    observed_bigrams = {
        (a, b) for s in sents for a, b in zip(s, s[1:])
    }
    so_zero_ctx = sorted(observed_bigrams - set(so_cpt.keys()))

    fo_gen = [" ".join(first_order.generate(fo_cpt, "sample", seed=i))
              for i in range(50)]
    so_gen = [" ".join(second_order.generate(so_cpt, "sample", seed=i))
              for i in range(50)]

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 60)
    emit("First-order vs second-order comparison")
    emit("=" * 60)
    emit(f"vocabulary size: {len(vocab)}")
    emit(f"first-order  distinct parameters (nonzero entries): {fo_params}")
    emit(f"second-order distinct parameters (nonzero entries): {so_params}")
    emit(f"first-order  contexts with no observed continuation:  {len(fo_zero_ctx)}")
    emit(f"second-order observed bigrams with no continuation:   {len(so_zero_ctx)}")
    emit(f"first-order  sentence diversity (50 samples): {diversity(fo_gen):.2f}")
    emit(f"second-order sentence diversity (50 samples): {diversity(so_gen):.2f}")

    emit("\nFirst-order examples:")
    for s in fo_gen[:6]:
        emit(f"  {s}")
    emit("\nSecond-order examples:")
    for s in so_gen[:6]:
        emit(f"  {s}")

    (here / "outputs" / "comparison.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
