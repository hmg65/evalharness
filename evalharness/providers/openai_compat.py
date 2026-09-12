"""OpenAI-compatible chat-completions client.

Works against any endpoint that speaks POST {base_url}/chat/completions with
an Authorization: Bearer key - OpenAI, Azure OpenAI, Together, Groq, Fireworks,
OpenRouter, a local vLLM/Ollama server, etc. Keys come from environment
variables only; nothing secret is ever written to the config or the logs.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from .base import ModelResponse, Timer


class OpenAICompatClient:
    def __init__(
        self,
        name: str,
        model_id: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self._headers = {"Content-Type": "application/json", **(extra_headers or {})}

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set "
                f"(needed for model '{self.name}'). Keys live in the environment, "
                "never in config files - see .env.example."
            )
        return key

    def complete(self, prompt: str, params: dict[str, Any]) -> ModelResponse:
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            **params,
        }
        headers = {**self._headers, "Authorization": f"Bearer {self._api_key()}"}
        timer = Timer()
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ModelResponse(
            text=choice["message"]["content"],
            latency_s=timer.elapsed(),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            model_id=data.get("model", self.model_id),
            raw={"finish_reason": choice.get("finish_reason")},
        )
