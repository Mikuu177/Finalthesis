from __future__ import annotations
import os
from typing import Dict, Any

from .openai_adapter import OpenAIAdapter
from .openai_compat_adapter import OpenAICompatAdapter


def build_adapter(name: str, cfg: Dict[str, Any]):
    provider = cfg.get('provider')
    model = cfg.get('model')
    api_key_env = cfg.get('api_key_env')
    # Try provider-specific env first, then fallback to OPENAI_API_KEY for safety
    api_key = os.getenv(api_key_env) if api_key_env else None
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')

    if provider == 'openai':
        return OpenAIAdapter(name=name, model=model, api_key=api_key)
    elif provider == 'openai_compatible':
        base_url = cfg.get('base_url')
        default_headers = cfg.get('default_headers')
        default_extra_body = cfg.get('default_extra_body')
        if not api_key:
            raise RuntimeError(f"API key not found for model '{name}'. Ensure {api_key_env} or OPENAI_API_KEY is set.")
        return OpenAICompatAdapter(
            name=name,
            model=model,
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            default_extra_body=default_extra_body,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
