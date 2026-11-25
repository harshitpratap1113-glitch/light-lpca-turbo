"""
Universal Zero-Markup BYOK LLM Router with Async SSE Token Streaming.
Supports Groq, Google Gemini, OpenAI, Anthropic Claude, DeepSeek, OpenRouter, and Local Ollama.
Automatically extracts <memory_update> tags into SovereignVault FTS5 memory.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from .security import redact_secrets
from .vault import SovereignVault


PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "groq": {
        "label": "Groq (Ultra-Fast Free Tier)",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "openai/gpt-oss-120b",
        "env_var": "GROQ_API_KEY",
        "style": "openai",
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "default_model": "gemini-2.0-flash",
        "env_var": "GEMINI_API_KEY",
        "style": "gemini",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
        "style": "openai",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "default_model": "deepseek-chat",
        "env_var": "DEEPSEEK_API_KEY",
        "style": "openai",
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "env_var": "OPENROUTER_API_KEY",
        "style": "openai",
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1/messages",
        "default_model": "claude-3-5-haiku-latest",
        "env_var": "ANTHROPIC_API_KEY",
        "style": "anthropic",
    },
    "ollama": {
        "label": "Ollama (100% Local Offline)",
        "base_url": "http://127.0.0.1:11434/api/chat",
        "default_model": "llama3.2",
        "env_var": "OLLAMA_HOST",
        "style": "ollama",
    },
}

MEMORY_PATTERN = re.compile(r"<memory_update>(.*?)</memory_update>", re.DOTALL | re.IGNORECASE)


def extract_and_store_memories(text: str, vault: Optional[SovereignVault], source: str = "chat") -> Tuple[str, List[str]]:
    """Extracts <memory_update> blocks, stores each line in FTS5 vault, and strips tags from display text."""
    extracted: List[str] = []
    for match in MEMORY_PATTERN.findall(text):
        for line in match.splitlines():
            clean = line.strip(" -*\t")
            if len(clean) >= 6:
                extracted.append(clean)
                if vault:
                    vault.add_memory(clean, category="cognitive", source=source)
    cleaned_text = MEMORY_PATTERN.sub("", text).strip()
    return cleaned_text, extracted


class BYOKRouter:
    """Routes chat and research synthesis directly to user-supplied LLM providers with zero proxy markup."""

    def __init__(self, vault: Optional[SovereignVault] = None, timeout: float = 30.0):
        self.vault = vault
        self.timeout = timeout

    def get_active_config(self) -> Dict[str, str]:
        saved = self.vault.get_setting("active_llm", {}) if self.vault else {}
        provider = saved.get("provider", "groq")
        preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["groq"])
        api_key = saved.get("api_key") or os.environ.get(preset["env_var"], "")
        return {
            "provider": provider,
            "model": saved.get("model") or preset["default_model"],
            "api_key": api_key,
            "base_url": saved.get("base_url") or preset["base_url"],
            "style": preset["style"],
        }

    def list_providers_status(self) -> List[Dict[str, Any]]:
        saved_keys = self.vault.get_setting("provider_keys", {}) if self.vault else {}
        active = self.get_active_config()["provider"]
        out = []
        for pid, meta in PROVIDER_PRESETS.items():
            key_val = saved_keys.get(pid) or os.environ.get(meta["env_var"], "")
            out.append(
                {
                    "id": pid,
                    "label": meta["label"],
                    "default_model": meta["default_model"],
                    "configured": bool(key_val) or pid == "ollama",
                    "active": pid == active,
                }
            )
        return out

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Yields text chunks from the chosen BYOK provider, or a local cognitive fallback if unconfigured."""
        cfg = self.get_active_config()
        prov = provider or cfg["provider"]
        preset = PROVIDER_PRESETS.get(prov, PROVIDER_PRESETS["groq"])
        mdl = model or (preset["default_model"] if provider else cfg["model"])
        saved_keys = self.vault.get_setting("provider_keys", {}) if self.vault else {}
        key = api_key or saved_keys.get(prov) or os.environ.get(preset["env_var"], "") or cfg["api_key"]

        if prov != "ollama" and not key:
            user_q = messages[-1]["content"] if messages else ""
            memories = self.vault.search_memories(user_q, limit=4) if self.vault else []
            mem_note = "\n".join(f"- [{m['category']}] {m['fact']}" for m in memories) if memories else "No prior memories matched."
            yield (
                "No external BYOK API key is configured for `" + prov + "` yet (running in Local Sovereign Mode).\n\n"
                "**Matched FTS5 Vault Memories:**\n" + mem_note + "\n\n"
                "Tip: Set `" + preset["env_var"] + "` or paste your free Groq/Gemini key in Settings to unlock live streaming synthesis."
            )
            return

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            style = preset["style"]
            if style == "openai":
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                payload = {"model": mdl, "messages": messages, "stream": True, "temperature": 0.3, "max_tokens": 1024}
                async with client.stream("POST", preset["base_url"], headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield redact_secrets(f"[{prov} HTTP {resp.status_code}]: {body.decode('utf-8', 'ignore')[:300]}")
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta
                        except Exception:
                            continue

            elif style == "gemini":
                url = f"{preset['base_url']}/{mdl}:streamGenerateContent?alt=sse&key={key}"
                contents = [
                    {
                        "role": "user" if m["role"] in ("user", "system") else "model",
                        "parts": [{"text": m["content"]}],
                    }
                    for m in messages
                ]
                async with client.stream("POST", url, json={"contents": contents}) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"[gemini HTTP {resp.status_code}]: {body.decode('utf-8', 'ignore')[:300]}"
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            obj = json.loads(line[6:])
                            for cand in obj.get("candidates", []):
                                for part in cand.get("content", {}).get("parts", []):
                                    if part.get("text"):
                                        yield part["text"]
                        except Exception:
                            continue

            elif style == "anthropic":
                headers = {
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                sys_prompt = "\n".join(m["content"] for m in messages if m["role"] == "system")
                chat_msgs = [m for m in messages if m["role"] != "system"]
                payload: Dict[str, Any] = {"model": mdl, "max_tokens": 1536, "messages": chat_msgs, "stream": True}
                if sys_prompt:
                    payload["system"] = sys_prompt
                async with client.stream("POST", preset["base_url"], headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"[anthropic HTTP {resp.status_code}]: {body.decode('utf-8', 'ignore')[:300]}"
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            ev = json.loads(line[6:])
                            if ev.get("type") == "content_block_delta":
                                txt = ev.get("delta", {}).get("text")
                                if txt:
                                    yield txt
                        except Exception:
                            continue

            elif style == "ollama":
                payload = {"model": mdl, "messages": messages, "stream": True}
                async with client.stream("POST", preset["base_url"], json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            ev = json.loads(line)
                            txt = ev.get("message", {}).get("content")
                            if txt:
                                yield txt
                        except Exception:
                            continue
