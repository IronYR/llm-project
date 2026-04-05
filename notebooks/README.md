# Notebooks

Inspired by the structure in [geetu040/nust-bank-chatbot](https://github.com/geetu040/nust-bank-chatbot) (`FineTuning.ipynb`, `Bechmarking.ipynb`), but using **this repository’s** stack: Qwen2.5-Instruct + PEFT/TRL + Qdrant + `sentence-transformers`.

| Notebook | Purpose |
|----------|---------|
| [01_data_preparation.ipynb](01_data_preparation.ipynb) | Explore `knowledge_base.json` and SFT row format |
| [02_lora_finetuning.ipynb](02_lora_finetuning.ipynb) | LoRA supervised fine-tuning (same logic as `train_finetune.py`) |
| [03_retrieval_benchmark.ipynb](03_retrieval_benchmark.ipynb) | Retrieval top-1 / top-k match rates vs `evaluation/run_eval.py` |

## Local Jupyter

From the project root: `pip install -r requirements.txt` (includes `jupyter` / `ipykernel`). Open a notebook; the first code cell skips cloning when not on Colab.

Run `python ingest.py` first so `data/processed/knowledge_base.json` (and Qdrant for `03`) exist—unless you use the Colab upload path below.

## Google Colab

Each notebook starts with an **Open in Colab** badge. **Before pushing to GitHub**, replace the placeholder **`YOUR_GITHUB_USERNAME`** in:

1. The badge URL in the first markdown cell (three places: one per notebook file).
2. The `GIT_REPO = "https://github.com/YOUR_GITHUB_USERNAME/llm-project.git"` line in the first code cell.

Use your real GitHub username or org and repo name if the fork is not called `llm-project`.

**Typical Colab flow**

1. Push this repo to GitHub (notebooks must live there for the Colab badge to open the right file).
2. Open the notebook from GitHub (badge) or **File → Upload notebook** and upload the `.ipynb` (then you must still run the clone cell so `src/` exists).
3. **Runtime → Change runtime type → GPU** (recommended for `02_lora_finetuning.ipynb`).
4. Run the bootstrap cell: clones into `/content/llm-project` and `pip install -r requirements.txt`.
5. Run `ensure_knowledge_base`: on Colab, upload the course **XLSX** + **JSON** when prompted; `ingest.py` runs and builds `knowledge_base.json` and the Qdrant index.

Helpers live in [`src/colab_setup.py`](../src/colab_setup.py) (`ensure_knowledge_base`, `is_colab`).

**Saving LoRA weights:** Colab disks are ephemeral. After training, copy `models/lora_nust_bank` to [Google Drive](https://colab.research.google.com/notebooks/io.ipynb) (mount Drive in a cell) or download the folder via the Colab file browser.

**Hugging Face:** If downloads are slow, set a [token](https://huggingface.co/settings/tokens) in Colab: `from huggingface_hub import login; login()` in a cell before loading the model.
