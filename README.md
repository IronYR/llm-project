# NUST Bank AI Assistant

An AI-powered customer service chatbot for NUST Bank, built with a **Retrieval-Augmented Generation (RAG)** pipeline. The assistant answers customer queries about bank products, accounts, loans, digital services, and more using the official NUST Bank knowledge base.

### Architecture Diagram
![WhatsApp Image 2026-03-07 at 3 05 33 PM](https://github.com/user-attachments/assets/f25c7b08-511c-48f5-b85c-17b760a0e9a9)
![WhatsApp Image 2026-03-07 at 3 05 33 PM (1)](https://github.com/user-attachments/assets/7ea7ac9f-2132-4034-b120-5f59de7bfb75)

## Setup & Installation

### Step 1 – Install Ollama and pull the model

```bash
# Install Ollama (macOS)
brew install ollama

# Start the Ollama server
ollama serve

# Pull the model (in a new terminal)
ollama pull qwen2.5:3b
```

### Step 2 – Create a Python virtual environment

```bash
cd llm-project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


### Step 3 – Build the knowledge base index

```bash
python ingest.py
```

This will:
1. Parse all 35 sheets of `NUST Bank-Product-Knowledge.xlsx`
2. Parse `funds_transfer_app_features_faq.json`
3. Clean, anonymize, and deduplicate documents
4. Build the Qdrant vector index at `data/qdrant_store/`

### Step 4 – Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

All settings live in `config.yaml`:

---


