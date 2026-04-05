import logging
import threading
from pathlib import Path
from typing import Generator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

logger = logging.getLogger(__name__)


def _device_map() -> str | dict:
    if torch.cuda.is_available():
        return "auto"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"": "mps"}
    return {"": "cpu"}


def _dtype():
    if torch.cuda.is_available():
        return torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _load_model_and_tokenizer(config: dict):
    ft = config.get("finetune", {})
    llm = config.get("llm", {})
    model_id = ft.get("base_model_id", llm.get("hf_model_id", "Qwen/Qwen2.5-3B-Instruct"))
    lora_dir = Path(ft.get("output_dir", "./models/lora_nust_bank"))
    use_lora = llm.get("use_lora", True)
    has_lora = use_lora and lora_dir.is_dir() and (lora_dir / "adapter_config.json").exists()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = llm.get("load_in_4bit", True) and torch.cuda.is_available()
    if quant:
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
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_dtype(),
            device_map=_device_map(),
            trust_remote_code=True,
        )

    if has_lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(lora_dir))
        logger.info("Loaded LoRA adapter from %s", lora_dir)

    model.eval()
    return model, tokenizer


class HFChatLLM:
    def __init__(self, config: dict) -> None:
        self.config = config
        llm = config.get("llm", {})
        self.max_new_tokens = llm.get("max_tokens", 600)
        self.temperature = llm.get("temperature", 0.3)
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._tokenizer = _load_model_and_tokenizer(self.config)

    def is_available(self) -> bool:
        try:
            self._ensure_loaded()
            return self._model is not None
        except Exception as e:
            logger.warning("HF model unavailable: %s", e)
            return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        dev = next(self._model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream_generate(
        self, system_prompt: str, user_prompt: str
    ) -> Generator[str, None, None]:
        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        dev = next(self._model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": max(self.temperature, 1e-5),
            "pad_token_id": self._tokenizer.pad_token_id,
            "streamer": streamer,
        }

        thread = threading.Thread(target=lambda: self._model.generate(**gen_kwargs))
        thread.start()
        for token in streamer:
            yield token
        thread.join()
