"""Minimal OpenAI Responses API wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from discoverllm.llm.adapters.http import ProviderAPIError, post_json
from discoverllm.llm.base import LLMAdapter, LLMRequest, LLMResponse
from discoverllm.types import ConversationMessage

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1/responses"


@dataclass(slots=True)
class OpenAIResponsesAdapter(LLMAdapter):
    """LLM adapter backed by OpenAI's Responses API."""

    api_key: str
    model: str
    base_url: str = DEFAULT_OPENAI_BASE_URL
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key.strip()
        self.model = self.model.strip()
        self.base_url = self.base_url.strip()
        self.timeout_seconds = float(self.timeout_seconds)
        if not self.api_key or not self.model:
            raise ValueError("api_key and model are required.")

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [self._message_to_input_item(message) for message in request.messages],
            "temperature": request.temperature,
        }
        if request.system_prompt is not None:
            payload["instructions"] = request.system_prompt
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens

        response_payload = post_json(
            url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key.strip()}"},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        text = extract_openai_text(response_payload)
        usage = response_payload.get("usage", {})
        prompt_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        return LLMResponse(
            text=text,
            raw_output=response_payload,
            model=str(response_payload.get("model", self.model)),
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=(
                completion_tokens if isinstance(completion_tokens, int) else None
            ),
        )

    def _message_to_input_item(self, message: ConversationMessage) -> dict[str, Any]:
        content_type = "input_text"
        if message.role == "assistant":
            content_type = "output_text"
        return {
            "role": message.role,
            "content": [{"type": content_type, "text": message.content}],
        }


def extract_openai_text(payload: dict[str, Any]) -> str:
    """Extract assistant text from a raw Responses API payload."""

    if not isinstance(payload, dict):
        raise ProviderAPIError("OpenAI response payload was not a JSON object.")
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if not isinstance(output, list):
        raise ProviderAPIError("OpenAI response payload is missing output text.")

    fragments: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content_items = item.get("content")
        if not isinstance(content_items, list):
            continue
        for content_item in content_items:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") not in ("output_text", "text"):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                fragments.append(text.strip())
    if fragments:
        return "\n".join(fragments)
    raise ProviderAPIError("OpenAI response payload did not contain text output.")
