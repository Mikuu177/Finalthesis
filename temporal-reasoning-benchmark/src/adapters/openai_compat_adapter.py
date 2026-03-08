from __future__ import annotations
from typing import Any, Dict, List, Optional
from openai import OpenAI

from .base_adapter import BaseAdapter, GenerateResult


class OpenAICompatAdapter(BaseAdapter):
    """
    OpenAI-compatible chat completions adapter.
    Works with providers exposing OpenAI-style endpoints via base_url.
    """
    def __init__(
        self,
        name: str,
        model: str,
        api_key: Optional[str],
        base_url: str,
        default_headers: Optional[Dict[str, str]] = None,
        default_extra_body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, model=model, api_key=api_key, base_url=base_url)
        # Note: openai>=1.0 constructor supports base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers or None)
        self.default_extra_body = default_extra_body or {}

    def _do_generate(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> GenerateResult:
        kwargs: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': params.get('temperature', 0.0),
            'top_p': params.get('top_p', 1.0),
            'max_tokens': params.get('max_tokens', 512),
            'timeout': params.get('timeout', 60),
        }
        extra_headers = params.get('extra_headers')
        if isinstance(extra_headers, dict) and extra_headers:
            kwargs['extra_headers'] = extra_headers
        extra_body = {}
        if self.default_extra_body:
            extra_body.update(self.default_extra_body)
        req_extra_body = params.get('extra_body')
        if isinstance(req_extra_body, dict) and req_extra_body:
            extra_body.update(req_extra_body)
        if extra_body:
            kwargs['extra_body'] = extra_body
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




