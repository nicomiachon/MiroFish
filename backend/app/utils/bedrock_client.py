"""
AWS Bedrock Converse API client.
Wraps responses to match the OpenAI SDK interface so consuming code doesn't need to change.
"""

import json
import requests
from typing import Optional, Dict, Any, List
from ..config import Config


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, message_content, finish_reason):
        self.message = _Message(message_content)
        self.finish_reason = finish_reason


class _ChatCompletion:
    def __init__(self, choices):
        self.choices = choices


class BedrockClient:
    """AWS Bedrock client using the Converse API with ABSK key auth."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.BEDROCK_API_KEY
        self.region = region or Config.AWS_REGION
        self.model = model or Config.CLAUDE_MODEL

        if not self.api_key:
            raise ValueError("BEDROCK_API_KEY not configured")

        self.base_url = f"https://bedrock-runtime.{self.region}.amazonaws.com"

    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> _ChatCompletion:
        """Call Bedrock Converse API, return OpenAI-compatible response object."""

        # Convert OpenAI message format to Bedrock format
        bedrock_messages = []
        system_text = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_text = content
                # If JSON mode requested, inject instruction into system prompt
                if response_format and response_format.get("type") == "json_object":
                    system_text += "\n\nIMPORTANT: You MUST respond with valid JSON only. No markdown code fences, no explanatory text before or after the JSON. Output pure JSON."
            else:
                bedrock_messages.append({
                    "role": role,
                    "content": [{"text": content}]
                })

        # If JSON mode but no system message, create one
        if response_format and response_format.get("type") == "json_object" and not system_text:
            system_text = "You MUST respond with valid JSON only. No markdown code fences, no explanatory text before or after the JSON. Output pure JSON."

        url = f"{self.base_url}/model/{model}/converse"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens
            }
        }

        if system_text:
            body["system"] = [{"text": system_text}]

        resp = requests.post(url, headers=headers, json=body, timeout=300)

        if resp.status_code != 200:
            raise Exception(f"Bedrock API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()

        # Extract content
        output = data.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        text = content_blocks[0]["text"] if content_blocks else ""

        # Track token usage
        usage = data.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        from .token_tracker import tracker
        tracker.add(input_tokens, output_tokens)

        # Map stop reason
        stop_reason = data.get("stopReason", "end_turn")
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "content_filtered": "content_filter",
        }
        finish_reason = finish_reason_map.get(stop_reason, "stop")

        return _ChatCompletion(
            choices=[_Choice(text, finish_reason)]
        )
