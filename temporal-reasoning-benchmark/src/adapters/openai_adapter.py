from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI

from .base_adapter import BaseAdapter, GenerateResult


class OpenAIAdapter(BaseAdapter):
    def __init__(self, name: str, model: str, api_key: Optional[str]):
        super().__init__(name=name, model=model, api_key=api_key, base_url=None)
        self.client = OpenAI(api_key=api_key)

    def _do_generate(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> GenerateResult:
        # Map generic params to OpenAI
        kwargs: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': params.get('temperature', 0.0),
            'top_p': params.get('top_p', 1.0),
            'max_tokens': params.get('max_tokens', 512),
            'timeout': params.get('timeout', 60),
        }
        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ''
        usage = getattr(resp, 'usage', None)
        if usage:
            usage = {
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
            }
        return GenerateResult(text=text, latency=0.0, usage=usage, error=None)






