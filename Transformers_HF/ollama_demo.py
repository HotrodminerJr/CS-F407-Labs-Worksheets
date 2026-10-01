"""Standalone mirror of Run_Ollama.ipynb.

Prerequisite (one-time):
    curl -fsSL https://ollama.com/install.sh | sh    # Linux
    ollama pull mistral                               # ~4 GB

Then:
    python ollama_demo.py "Who is Sir Isaac Newton?"

The ChatPromptTemplate + OllamaLLM wiring is the same as the notebook; this
file just makes it runnable from a terminal and captures the output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM
except ImportError:
    print(
        "Missing dependencies. Install with:\n"
        "    pip install langchain-ollama langchain-core\n"
        "You also need the Ollama server running locally; see the file header.",
        file=sys.stderr,
    )
    sys.exit(1)


TEMPLATE = """Question: {question}

Answer: Let's think step by step."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?",
                    default="Who is Sir Isaac Newton?")
    ap.add_argument("--model", default="mistral",
                    help="any model pulled with `ollama pull <name>`")
    args = ap.parse_args()

    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    model = OllamaLLM(model=args.model)
    chain = prompt | model

    reply = chain.invoke({"question": args.question})
    print(f"Q: {args.question}\n\nA:\n{reply}\n")

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ollama_reply.txt").write_text(
        f"model: {args.model}\nQ: {args.question}\n\nA:\n{reply}\n"
    )


if __name__ == "__main__":
    main()
