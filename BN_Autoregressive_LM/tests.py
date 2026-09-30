"""Probability-invariant tests for the first- and second-order models.

Part VII of the worksheet asks: for every context w,
    sum_v P(v | w) = 1.
We check this for both models and print any context that fails.
"""

from __future__ import annotations

from pathlib import Path

import first_order
import second_order

TOL = 1e-9


def check(cpt, label: str) -> list[str]:
    lines = [f"[{label}]"]
    failed = 0
    for ctx, dist in sorted(cpt.items(), key=lambda kv: str(kv[0])):
        total = sum(dist.values())
        ok = abs(total - 1.0) < TOL
        marker = "OK " if ok else "BAD"
        lines.append(f"  {marker}  sum P(v | {ctx!r}) = {total:.9f}")
        if not ok:
            failed += 1
    lines.append(f"  -> {failed} failing context(s)")
    return lines


def main() -> None:
    here = Path(__file__).parent
    sents = first_order.load_sentences(here / "dataset.txt")

    fo_cpt = first_order.build_cpt(first_order.count_transitions(sents))
    so_cpt = second_order.build_cpt(second_order.count_triples(sents))

    lines: list[str] = []
    lines += check(fo_cpt, "first-order  P(X_t | X_{t-1})")
    lines.append("")
    lines += check(so_cpt, "second-order P(X_t | X_{t-2}, X_{t-1})")

    text = "\n".join(lines) + "\n"
    print(text)
    (here / "outputs" / "test_results.txt").write_text(text)


if __name__ == "__main__":
    main()
