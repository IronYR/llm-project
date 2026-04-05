# NUST Bank AI Assistant

AI customer service assistant for NUST Bank using **Retrieval-Augmented Generation (RAG)** on the official product knowledge base, with optional **LoRA fine-tuning** (PEFT) on the same Q&A data and a **LangGraph**-based query workflow.

## Architecture

1. **Data pipeline** (`ingest.py`, `src/data_ingestion.py`): parse XLSX + JSON, anonymize PII, build `knowledge_base.json`, embed with `sentence-transformers` / `all-MiniLM-L6-v2`, index in **Qdrant**.
2. **Query pipeline** (`src/rag_graph.py`): **LangGraph** graph with nodes: input guardrails, retrieval, conditional routing (off-topic / no context / generate), generation, output checks. Implemented in `RAGGraphRunner`; `src/rag_pipeline.py` exposes the same API to the UI.
3. **LLM backends** (`config.yaml` `llm.backend`):
  - `ollama`: default, `qwen2.5:3b` via local Ollama API.
  - `hf`: Hugging Face `Qwen/Qwen2.5-3B-Instruct` with optional LoRA adapter from `finetune.output_dir` (`src/hf_llm.py`, `src/llm_backend.py`).
4. **Fine-tuning** (`train_finetune.py`, shared logic in `src/finetune_lib.py`): LoRA on top of the base instruct model using **PEFT** + **bitsandbytes** 4-bit when CUDA is available; training rows built from `knowledge_base.json` with the same system prompt as RAG (`src/prompts.py`). Step-by-step **Jupyter** workflows live under [`notebooks/`](notebooks/README.md) (data prep, LoRA training, retrieval benchmark), similar in spirit to [nust-bank-chatbot](https://github.com/geetu040/nust-bank-chatbot)’s notebooks but using this repo’s tools.

### Note

Our proposal mentioned Unsloth and spaCy/NLTK. The codebase instead uses PEFT + TRL (and bitsandbytes for 4-bit when training on CUDA) in `train_finetune.py`—the same LoRA/quantization ideas, without the Unsloth dependency. For ingestion, text is cleaned and anonymized with regex in `src/data_ingestion.py` rather than spaCy, as the bank sources were mostly structured sheets and FAQs, so heavy linguistic tokenization was not required to meet the preprocessing goals.

## Setup

### Dependencies

- Python 3.10+
- For default inference: [Ollama](https://ollama.com) with `qwen2.5:3b`
- For training and HF inference: NVIDIA GPU with CUDA recommended (4-bit LoRA training)

```bash
cd llm-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ingest and index

Place `NUST Bank-Product-Knowledge.xlsx` and `funds_transfer_app_features_faq.json` under `data/raw/`, then:

```bash
python ingest.py
```

### Run the app (Ollama backend)

```bash
ollama serve
ollama pull qwen2.5:3b
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Fine-tune LoRA

Requires CUDA for 4-bit training (`finetune.use_4bit: true`). On macs without CUDA, set `finetune.use_4bit: false`; leave `finetune.use_mps: false` (default) so loading uses CPU. Set `use_mps: true` only if you accept the risk of load failures.

**CLI:**

```bash
python train_finetune.py
```

**Notebooks** (interactive, same training code via `src/finetune_lib.py`): install Jupyter if needed (`pip install jupyter ipykernel`), then open `notebooks/02_lora_finetuning.ipynb` from the project root. **Google Colab:** each notebook under `notebooks/` has an *Open in Colab* badge; clone your fork, install deps, and (on Colab) upload the course dataset when prompted—see [`notebooks/README.md`](notebooks/README.md).

Adapter is written to `finetune.output_dir` (default `./models/lora_nust_bank/`).

To serve answers with the fine-tuned weights via Hugging Face:

```yaml
llm:
  backend: hf
  use_lora: true
```

Ensure `finetune.output_dir` contains `adapter_config.json` after training.

### Retrieval evaluation

```bash
python evaluation/run_eval.py
python evaluation/run_eval.py --limit 50
```

Reports top-1 and top-k match rates against stored questions in the knowledge base.

## Configuration

Key keys in `config.yaml`:


| Section            | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| `llm.backend`      | `ollama` or `hf`                                     |
| `llm.use_lora`     | Load LoRA from `finetune.output_dir` when using `hf` |
| `llm.load_in_4bit` | 4-bit load for HF path on CUDA                       |
| `finetune.*`       | LoRA ranks, epochs, batch size, output directory     |
| `retrieval.*`      | Qdrant top-k and score threshold                     |


## Project layout

```
ingest.py                 # Build KB + Qdrant index
train_finetune.py         # LoRA fine-tuning CLI (PEFT + bitsandbytes)
notebooks/                # Jupyter: data prep, LoRA SFT, retrieval benchmark
evaluation/run_eval.py    # Retrieval metrics
app.py                    # Streamlit UI
config.yaml
requirements-colab.txt    # Colab: deps without torch (avoids CUDA wheel mismatch)
src/finetune_lib.py       # Shared SFT helpers (CLI + notebooks)
src/colab_setup.py        # Colab ingest / knowledge-base check for notebooks
src/data_ingestion.py
src/embeddings.py
src/guardrails.py
src/prompts.py
src/llm_client.py         # Ollama
src/hf_llm.py             # HF + optional LoRA
src/llm_backend.py        # Select backend
src/rag_graph.py          # LangGraph RAG
src/rag_pipeline.py     # Thin wrapper
```

## References

- Qwen2.5 technical report.
- Qdrant vector search documentation.
- LangGraph documentation (LangChain).

