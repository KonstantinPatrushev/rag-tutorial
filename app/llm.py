"""Клиент LLM (OpenAI-совместимый API).

Поддерживает облачный OpenAI и любой OpenAI-совместимый endpoint
(локальный vLLM/Ollama/llama.cpp, прокси и т.п.) через OPENAI_BASE_URL.
Ключ и параметры — из окружения (см. app/config.py и .env.example).

Если ключа нет — llm_enabled() == False, и generator переключается на
extractive demo-режим. Так проект запускается на чистой машине без секретов.
"""

from __future__ import annotations

from app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        _CLIENT = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _CLIENT


def chat(system: str, user: str, model: str = LLM_MODEL) -> str:
    """Один синхронный вызов chat-completions. Возвращает текст ответа."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()
