from __future__ import annotations
import time
import math
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..io_utils import cache_get, cache_put


@dataclass
class GenerateResult:
    text: str
    latency: float
    usage: Dict[str, Any] | None
    error: str | None = None


class BaseAdapter:
    def __init__(self, name: str, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def _do_generate(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> GenerateResult:
        raise NotImplementedError

    def generate(self, messages: List[Dict[str, str]], params: Dict[str, Any], cache_dir: str | None = None,
                 cache_key_extra: str | None = None, retries: int = 3, backoff_base: float = 1.5) -> GenerateResult:
        # build fingerprint for caching
        fingerprint = {
            'model': self.model,
            'messages': messages,
            'params': params,
            'extra': cache_key_extra or ''
        }
        import json
        fp_str = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True)

        if cache_dir:
            cached = cache_get(cache_dir, self.name, fp_str)
            if cached:
                return GenerateResult(
                    text=cached.get('text', ''),
                    latency=cached.get('latency', 0.0),
                    usage=cached.get('usage'),
                    error=cached.get('error')
                )

        attempt = 0
        last_err: str | None = None
        while attempt <= retries:
            start = time.time()
            try:
                result = self._do_generate(messages, params)
                end = time.time()
                result.latency = end - start
                if cache_dir:
                    cache_put(cache_dir, self.name, fp_str, {
                        'text': result.text,
                        'latency': result.latency,
                        'usage': result.usage,
                        'error': result.error,
                    })
                return result
            except Exception as e:
                last_err = str(e)
                # exponential backoff
                if attempt == retries:
                    break
                sleep_s = backoff_base ** attempt + (0.1 * attempt)
                time.sleep(sleep_s)
                attempt += 1
        return GenerateResult(text='', latency=0.0, usage=None, error=last_err or 'Unknown error')






