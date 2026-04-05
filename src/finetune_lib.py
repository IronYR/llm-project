"""
Shared helpers for LoRA SFT (PEFT + TRL + bitsandbytes).
Used by `train_finetune.py` and `notebooks/02_lora_finetuning.ipynb`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig

from src.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset_rows(processed_dir: str, kb_file: str) -> list[dict]:
    kb_path = Path(processed_dir) / kb_file
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)
    rows: list[dict] = []
    for doc in kb.get("documents", []):
        q = (doc.get("question") or "").strip()
        a = (doc.get("answer") or "").strip()
        cat = doc.get("category", "")
        if not a:
            continue
        if q:
            instr = f"Product: {cat}\nQuestion: {q}"
        else:
            instr = f"Product: {cat}\nSummarize the following information."
        rows.append({"instruction": instr, "response": a})
    return rows


def build_sft_dataset(tokenizer: AutoTokenizer, rows: list[dict]) -> Dataset:
    text_rows: list[dict[str, str]] = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]
        text_rows.append(
            {
                "text": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            }
        )
    return Dataset.from_list(text_rows)


def load_tokenizer(model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, low_cpu_mem_usage=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model_for_training(model_id: str, ft: dict[str, Any]) -> AutoModelForCausalLM:
    use_4bit = ft.get("use_4bit", True) and torch.cuda.is_available()
    common_kw = dict(trust_remote_code=True, low_cpu_mem_usage=True)

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            **common_kw,
        )
        return prepare_model_for_kbit_training(model)

    if torch.cuda.is_available():
        dm, dt = "auto", torch.float16
    elif ft.get("use_mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dm, dt = {"": "mps"}, torch.float16
        logger.warning(
            "finetune.use_mps: true — if load fails with Invalid buffer size, set use_mps: false (CPU)."
        )
    else:
        dm, dt = {"": "cpu"}, torch.float16
        if not torch.cuda.is_available():
            logger.info(
                "No CUDA: training on CPU (default). For 4-bit LoRA use an NVIDIA GPU. "
                "MPS is off by default because large-model load often hits MPS allocator limits."
            )

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dt,
        device_map=dm,
        **common_kw,
    )


def build_lora_config(ft: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=ft.get("lora_r", 16),
        lora_alpha=ft.get("lora_alpha", 32),
        lora_dropout=ft.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=ft.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )


def build_sft_training_args(ft: dict[str, Any], output_dir: Path) -> SFTConfig:
    use_mps = bool(
        ft.get("use_mps")
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=ft.get("num_epochs", 2),
        per_device_train_batch_size=ft.get("batch_size", 1),
        gradient_accumulation_steps=ft.get("gradient_accumulation_steps", 8),
        learning_rate=ft.get("learning_rate", 2e-4),
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        fp16=bool(torch.cuda.is_available() or use_mps),
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        max_length=ft.get("max_seq_length", 2048),
        dataset_text_field="text",
        packing=False,
    )


def attach_lora(model: AutoModelForCausalLM, ft: dict[str, Any]) -> Any:
    lora = build_lora_config(ft)
    return get_peft_model(model, lora)
