from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class LLMConfig:
    # Paper-aligned defaults.
    chat_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = ""
    temperature: float = 0.3
    max_output_tokens: int = 1600


class LLMClient:
    """Thin OpenAI wrapper for chat JSON and embeddings."""

    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig()
        api_key = os.environ.get(self.cfg.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key in env var '{self.cfg.api_key_env}'. "
                "Set it before running DatasetSAGE."
            )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.cfg.base_url.strip():
            kwargs["base_url"] = self.cfg.base_url.strip()
        self.client = OpenAI(**kwargs)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(
            model=self.cfg.embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            temperature=self.cfg.temperature,
            response_format={"type": "json_object"},
            max_tokens=self.cfg.max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned invalid JSON: {content[:500]}") from exc
