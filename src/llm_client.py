import json
import logging
from typing import Generator

import requests

logger = logging.getLogger(__name__)

_OLLAMA_TIMEOUT = 120  # seconds


class OllamaClient:
    def __init__(self, config: dict) -> None:
        llm_cfg = config["llm"]
        self.model       = llm_cfg["model"]
        self.base_url    = llm_cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self.temperature = llm_cfg.get("temperature", 0.3)
        self.max_tokens  = llm_cfg.get("max_tokens", 600)
        self.fallback    = llm_cfg.get("fallback_model", "mistral:latest")


    def _chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

    def _options(self) -> dict:
        return {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }


    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            tags = resp.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in tags]
            target = self.model.split(":")[0]
            return target in model_names
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def generate(self, system_prompt: str, user_prompt: str, model: str | None = None) -> str:
        target_model = model or self.model
        payload = {
            "model":    target_model,
            "messages": self._build_messages(system_prompt, user_prompt),
            "stream":   False,
            "options":  self._options(),
        }

        try:
            resp = requests.post(
                self._chat_url(),
                json=payload,
                timeout=_OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Run: ollama serve\n"
                f"Then: ollama pull {self.model}"
            )
        except Exception as e:
            if target_model != self.fallback:
                logger.warning("Model %s failed (%s), trying fallback %s", target_model, e, self.fallback)
                return self.generate(system_prompt, user_prompt, model=self.fallback)
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> Generator[str, None, None]:
        target_model = model or self.model
        payload = {
            "model":    target_model,
            "messages": self._build_messages(system_prompt, user_prompt),
            "stream":   True,
            "options":  self._options(),
        }

        try:
            with requests.post(
                self._chat_url(),
                json=payload,
                stream=True,
                timeout=_OLLAMA_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.ConnectionError:
            yield (
                "\n\nCannot connect to Ollama. Run ollama serve and "
                f"ollama pull {self.model}"
            )
        except Exception as e:
            if target_model != self.fallback:
                logger.warning("Streaming with %s failed, retrying with %s", target_model, self.fallback)
                yield from self.stream_generate(system_prompt, user_prompt, model=self.fallback)
            else:
                yield f"\n\nLLM error: {e}"
