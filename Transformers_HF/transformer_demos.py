"""Standalone CLI mirror of AI_lab_transformers.ipynb.

Runs each of the three transformer families with a small, CPU-friendly model
so the demo is tractable without a GPU:

    encoder-only       : distilbert-base-uncased-finetuned-sst-2-english (sentiment)
    decoder-only       : gpt2                                            (text generation)
    encoder-decoder    : google-t5/t5-small                              (translation)

Usage:
    python transformer_demos.py                 # run all three
    python transformer_demos.py --family encoder
    python transformer_demos.py --family decoder
    python transformer_demos.py --family seq2seq

The notebook (AI_lab_transformers.ipynb) uses larger models that need a GPU
or significant disk; this script is the stdlib-size counterpart suitable for
reproducing the behaviour on a laptop CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from transformers import pipeline  # type: ignore
except ImportError:
    print(
        "The 'transformers' library is not installed in this environment.\n"
        "To run the demos, create a virtualenv and install:\n"
        "    pip install 'transformers[torch]' --index-url https://download.pytorch.org/whl/cpu\n"
        "On first run each pipeline will download its model (10-500 MB).",
        file=sys.stderr,
    )
    sys.exit(1)


OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


def demo_encoder() -> str:
    """Encoder-only: sentence classification (sentiment)."""
    clf = pipeline("sentiment-analysis",
                   model="distilbert-base-uncased-finetuned-sst-2-english")
    sentences = [
        "I love this new transformer lab, it finally makes attention click.",
        "Why does every notebook need a 15 GB model, I give up.",
        "The weather is fine.",
    ]
    lines = ["Encoder-only demo (DistilBERT, SST-2):"]
    for s in sentences:
        r = clf(s)[0]
        lines.append(f"  {s!r} -> {r['label']} ({r['score']:.3f})")
    return "\n".join(lines)


def demo_decoder() -> str:
    """Decoder-only: free-form continuation."""
    gen = pipeline("text-generation", model="gpt2")
    prompt = "The three families of transformer models are"
    out = gen(prompt, max_new_tokens=40, num_return_sequences=1,
              do_sample=False)[0]["generated_text"]
    return "Decoder-only demo (GPT-2, greedy):\n  " + out


def demo_seq2seq() -> str:
    """Encoder-decoder: English -> French with T5-small."""
    gen = pipeline("translation_en_to_fr", model="google-t5/t5-small")
    sentences = [
        "The cat is on the mat.",
        "Attention is all you need.",
        "I would like a cup of coffee.",
    ]
    lines = ["Encoder-decoder demo (T5-small, en->fr):"]
    for s in sentences:
        out = gen(s, max_new_tokens=40)[0]["translation_text"]
        lines.append(f"  EN: {s}\n  FR: {out}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("encoder", "decoder", "seq2seq", "all"),
                    default="all")
    args = ap.parse_args()

    chunks: list[str] = []
    if args.family in ("encoder", "all"):
        chunks.append(demo_encoder())
    if args.family in ("decoder", "all"):
        chunks.append(demo_decoder())
    if args.family in ("seq2seq", "all"):
        chunks.append(demo_seq2seq())

    text = "\n\n".join(chunks) + "\n"
    print(text)
    (OUT / "demo_output.log").write_text(text)
    print(f"saved outputs/demo_output.log")


if __name__ == "__main__":
    main()
