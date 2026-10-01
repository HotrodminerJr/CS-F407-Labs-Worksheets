# Lab — Transformers (HuggingFace) and Running an LLM Locally with Ollama

**Course:** CS F407 — Artificial Intelligence
**Lecture notes:** [`Transformer.pdf`](Transformer.pdf)

**Hands-on notebooks (verbatim from the lab):**

- [`AI_lab_transformers.ipynb`](AI_lab_transformers.ipynb) — one representative of each transformer family through HuggingFace `transformers`
- [`Run_Ollama.ipynb`](Run_Ollama.ipynb) — install Ollama locally and drive it with `langchain-ollama`

**Reproducible CLI mirrors (CPU-friendly, so you can run them on a laptop):**

- [`transformer_demos.py`](transformer_demos.py) — same three families with smaller models
- [`ollama_demo.py`](ollama_demo.py) — single-prompt Ollama + LangChain

```bash
pip install "transformers[torch]" --index-url https://download.pytorch.org/whl/cpu
python transformer_demos.py                 # all three families
python transformer_demos.py --family seq2seq
```

---

## 1 The Transformer — one architecture, three families

Original paper: Vaswani et al., *Attention Is All You Need* (2017). Introduced
for **machine translation** — one sequence in, another sequence out, no RNN,
no convolution, just stacked **self-attention** blocks.

Three families specialise the original encoder-decoder:

| Family | Representative | What it does well | Why |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa, DistilBERT | understand / classify / retrieve existing text | bidirectional self-attention: every position attends to every other, so each token's representation is informed by its full context |
| **Decoder-only** | GPT-2/3/4, Llama, Mistral | *generate* text / code / conversation | masked self-attention so position `t` only sees `≤ t`; makes next-token prediction a left-to-right factorisation |
| **Encoder-decoder** | T5, BART, BERT2BERT | translation, summarisation, any seq2seq | encoder reads the source bidirectionally; decoder generates target auto-regressively with **cross-attention** into the encoder output |

## 2 The six primitives every family uses

1. **Attention** — weighted sum over values `V` using similarity between queries
   `Q` and keys `K`:
   `Attention(Q,K,V) = softmax(Q Kᵀ / √dₖ) V`.
   `√dₖ` keeps the dot-product magnitudes well-scaled so the softmax does not
   saturate.
2. **Self-attention** — `Q, K, V` all come from the *same* sequence. Lets each
   token summarise the rest of the sentence into its own vector.
3. **Cross-attention** — `Q` from the decoder, `K` and `V` from the encoder.
   This is the bridge in encoder-decoder models that lets the decoder *look at*
   the source while generating the target.
4. **Multi-head attention** — run `h` smaller attention operations in parallel
   with different learned `W_Q, W_K, W_V` projections, concatenate, project
   back. Different heads can specialise on syntactic vs. semantic vs. positional
   patterns.
5. **Positional encoding** — attention itself is permutation-invariant. The
   original paper injects fixed sinusoids `PE(pos, 2i) = sin(pos / 10000^(2i/d))`;
   BERT learns absolute positions; T5/Llama use relative or rotary positions.
   Without this, "dog bites man" and "man bites dog" look identical.
6. **Masked attention** — a lower-triangular mask set to `−∞` before softmax,
   so position `t` cannot peek at positions `> t`. This is what makes a decoder
   auto-regressive during training (teacher-forcing on shifted targets).

Figure (from the lecture): encoder blocks on the left (self-attention → FFN),
decoder blocks on the right (masked self-attention → cross-attention → FFN),
with positional encoding added to both input embeddings.

## 3 Hands-on — what each notebook cell shows

### 3.1 Encoder-decoder — translation

```python
model = EncoderDecoderModel.from_pretrained("google/bert2bert_L-24_wmt_en_de")
# "Plants create energy through a process known as"
#   -> "Pflanzen erzeugen Energie durch einen so genannten Prozess."
```

```python
AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
# "translate English to French: Let's go have some wine, cause thats how we shine"
#   -> "Allons en déguster un peu de vin, car c'est ainsi que nous brillons"
```

```python
AutoModelForSeq2SeqLM.from_pretrained("rvv-karma/English2Hinglish-Flan-T5-Base")
# "life is crazy. Not me"  ->  "Life crazy haiNot me"
```

T5 is special because its *task prefix* ("translate English to French:", "summarize:")
is part of the input text, so one checkpoint handles many seq2seq tasks.

### 3.2 Decoder-only — generation

```python
pipeline("text-generation", model="gpt2")
# "The future of AI is" -> "...not yet clear, but it is likely to be worth
#                           exploring the potential benefits of artificial
#                           intelligence and how it might shape the world."
```

Decoder-only models have *no encoder*, so the "input" is just the prompt
prefix of the generation sequence. Everything you see as "generation" is
`model.generate(input_ids)` repeatedly sampling the next token from
`softmax(logits / T)`.

### 3.3 Encoder-only — classification

```python
pipeline("sentiment-analysis")  # defaults to distilbert SST-2
# "I'd want to kick you and ensure that you are hurt"
#   -> {"label": "NEGATIVE", "score": 0.974}
```

The encoder produces one vector per token; a tiny classification head over
the `[CLS]` token's vector produces the label logits. **No generation
happens** — the output space is `{POSITIVE, NEGATIVE}` here, not vocabulary.

## 4 Running an LLM locally — Ollama

Ollama packages open-source decoder-only LLMs (Llama, Mistral, Qwen,
Phi, …) as a local HTTP server, so you can run them on your own machine
instead of calling a cloud API. The notebook wires it to LangChain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

prompt = ChatPromptTemplate.from_template(
    "Question: {question}\n\nAnswer: Let's think step by step."
)
chain = prompt | OllamaLLM(model="mistral")
chain.invoke({"question": "Who is Sir Isaac Newton?"})
```

### Setup (not required for the notebook grading, but included for reproducibility)

```bash
# install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# pull a 7B model (first run is a few GB download)
ollama pull mistral

# verify
ollama list

# run from Python
pip install langchain-ollama
python ollama_demo.py "Explain self-attention in two sentences."
```

**Why run locally?** No API costs, no data leaves the machine, works offline,
reproducible across runs if you pin the model + a fixed seed. The *trade-off*
is quality — a 7B local model is noticeably weaker than frontier cloud models,
and generation is CPU-bound without a GPU.

## 5 Reflection

### What are the three families *really* for?

- **Need to understand text** (classify, retrieve, cluster, embed) → encoder-only.
- **Need to generate text** (chat, writing, code) → decoder-only.
- **Need to transform a sequence into another sequence** (translate,
  summarise, correct) → encoder-decoder. Also achievable by prompting a
  decoder-only LLM — "translate English to French: ...". The architecture
  is less deterministic of the capability than it used to be.

### Why does masked attention matter for training a decoder?

Teacher forcing shows the model the *whole* target sequence at once and asks
it to predict each next token. Without the lower-triangular mask the decoder
could trivially "predict" `t+1` by attending to `t+1`'s embedding at position
`t`. The mask forces an honest left-to-right factorisation.

### Why positional encoding?

Self-attention is a sum over positions; the operation itself has no notion of
order. Adding a position-dependent vector gives each embedding a positional
"watermark" so attention can learn position-sensitive patterns ("the verb comes
after the subject in English, but not always in German").

### Where the LLM-as-assistant pattern matters

The transformer *library* `transformers` already encapsulates most of the
engineering. The code in these notebooks is three lines of `pipeline(...)`;
the hard engineering decisions were made upstream by the model authors. The
same can't be said when you ask an LLM to **generate** code for a
probabilistic model from scratch — which is exactly what the companion lab
[`BN_LLM_Codegen`](../BN_LLM_Codegen/) does.
