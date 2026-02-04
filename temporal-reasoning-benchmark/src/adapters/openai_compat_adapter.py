from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI

from .base_adapter import BaseAdapter, GenerateResult


class OpenAICompatAdapter(BaseAdapter):
    """
    OpenAI-compatible chat completions adapter.
    Works with providers exposing OpenAI-style endpoints via base_url.
    """
    def __init__(self, name: str, model: str, api_key: Optional[str], base_url: str):
        super().__init__(name=name, model=model, api_key=api_key, base_url=base_url)
        # Note: openai>=1.0 constructor supports base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _do_generate(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> GenerateResult:
        kwargs: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': params.get('temperature', 0.0),
            'top_p': params.get('top_p', 1.0),
            'max_tokens': params.get('max_tokens', 512),
            # Some providers may ignore timeout here.
        }
        resp = self.client.chat.completions.create(**kwargs)
        # Some providers may return different shapes; be defensive
        try:
            text = resp.choices[0].message.content or ''
        except Exception:
            text = str(resp)
        usage = getattr(resp, 'usage', None)
        if usage:
            try:
                usage = {
                    'prompt_tokens': getattr(usage, 'prompt_tokens', None),
                    'completion_tokens': getattr(usage, 'completion_tokens', None),
                    'total_tokens': getattr(usage, 'total_tokens', None),
                }
            except Exception:
                usage = None
        return GenerateResult(text=text, latency=0.0, usage=usage, error=None)






