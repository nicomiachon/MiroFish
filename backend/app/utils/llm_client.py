"""
LLM client wrapper.
Supports OpenAI-compatible APIs and AWS Bedrock (Converse API with ABSK key).
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config


class LLMClient:
    """Unified LLM client supporting OpenAI and Bedrock providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        default_key, default_url, default_model = Config.get_llm_config()
        self.api_key = api_key or default_key
        self.base_url = base_url or default_url
        self.model = model or default_model
        self.provider = Config.LLM_PROVIDER

        if not self.api_key:
            raise ValueError("LLM API key not configured")

        if self.provider == 'bedrock':
            from .bedrock_client import BedrockClient
            self._bedrock = BedrockClient(
                api_key=self.api_key,
                region=Config.AWS_REGION,
                model=self.model
            )
        else:
            self._bedrock = None
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """Send a chat request and return the response text."""

        if self.provider == 'bedrock':
            response = self._bedrock.chat_completions_create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        else:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format
            response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        # Some models (e.g. MiniMax M2.5) include <think> tags — strip them
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Send a chat request and return parsed JSON."""
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # Clean markdown code block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from LLM: {cleaned_response}")
