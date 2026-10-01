# Transformers (HuggingFace) + Running an LLM Locally with Ollama

Lab for CS F407 covering:

- the three transformer families (encoder-only, decoder-only, encoder-decoder);
- the six attention primitives (self-, cross-, multi-head, positional, masked);
- hands-on HuggingFace `transformers` for sentiment, translation, generation;
- running a local open-source LLM via Ollama + LangChain.

## Files

| File | Role |
|---|---|
| [`Transformer.pdf`](Transformer.pdf) | Lecture notes |
| [`AI_lab_transformers.ipynb`](AI_lab_transformers.ipynb) | Original hands-on notebook (BERT2BERT, T5, GPT-2, DistilBERT) |
| [`Run_Ollama.ipynb`](Run_Ollama.ipynb) | Original Ollama + LangChain notebook |
| [`answers.md`](answers.md) | Full write-up — concepts, family comparison, reflection |
| [`transformer_demos.py`](transformer_demos.py) | CPU-friendly CLI mirror of the notebook |
| [`ollama_demo.py`](ollama_demo.py) | CLI mirror of the Ollama cell |

## Reproduce

```bash
# transformers demos (uses small CPU models)
pip install "transformers[torch]" --index-url https://download.pytorch.org/whl/cpu
python transformer_demos.py                 # runs all three families

# ollama demo
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
pip install langchain-ollama
python ollama_demo.py "Explain self-attention in two sentences."
```

## In one line

- **Encoder-only** → understand text (BERT).
- **Decoder-only** → generate text (GPT, Llama, Mistral).
- **Encoder-decoder** → transform one sequence into another (T5, BART, BERT2BERT).

All three are just stacks of attention + feed-forward blocks differing in
*which* attention mask is used.
