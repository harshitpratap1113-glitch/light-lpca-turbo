"""
LIGHT-LPCA Turbo v2.0 Local-First FastAPI Server.
Zero Cloudflare latency, zero 455KB JS bloat, SSE streaming chat, FTS5 Memory, and Shadow Brain control.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .byok_router import BYOKRouter, extract_and_store_memories
from .engine import TurboResearchAgent
from .security import SecurityHeadersMiddleware, SlidingWindowRateLimiter, get_client_ip, redact_secrets
from .shadow_brain import ShadowBrain
from .vault import SovereignVault


WEB_INDEX = Path(__file__).parent / "web" / "index.html"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="CORE_STREAM", max_length=64)
    provider: Optional[str] = Field(default=None, max_length=32)
    model: Optional[str] = Field(default=None, max_length=128)
    api_key: Optional[str] = Field(default=None, max_length=256)
    use_research: bool = False


class TestLLMRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None


class ResearchRequest(BaseModel):
    topic: str
    max_queries: int = 1
    max_pages: int = 2
    use_cache: bool = True


class ShadowToggleRequest(BaseModel):
    enabled: bool


class ShadowTaskRequest(BaseModel):
    title: str
    query: str
    interval_minutes: int = 60


class ProviderSaveRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    activate: bool = True


class MemoryAddRequest(BaseModel):
    fact: str
    category: str = "manual"
    importance: int = 7


def create_app(vault: Optional[SovereignVault] = None) -> FastAPI:
    v = vault or SovereignVault()
    router = BYOKRouter(vault=v)
    shadow = ShadowBrain(vault=v, pulse_seconds=45)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        bg_task = None
        is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
        if not is_serverless and v.get_setting("shadow_enabled", True):
            bg_task = asyncio.create_task(shadow.start_loop())
        try:
            yield
        finally:
            shadow.stop()
            if bg_task:
                bg_task.cancel()

    app = FastAPI(title="LIGHT-LPCA Turbo v2.0", version="2.0.0", lifespan=lifespan)
    app.add_middleware(SecurityHeadersMiddleware)
    rate_limiter = SlidingWindowRateLimiter(max_requests=25, window_seconds=60)
    app.state.vault = v
    app.state.router = router
    app.state.shadow = shadow
    app.state.rate_limiter = rate_limiter

    @app.get("/", response_class=HTMLResponse)
    async def home() -> str:
        if WEB_INDEX.exists():
            return WEB_INDEX.read_text(encoding="utf-8")
        return "<h1>LIGHT-LPCA Turbo v2.0</h1>"

    @app.get("/api/health")
    async def health(request: Request) -> Dict[str, Any]:
        ip = get_client_ip(request)
        return {
            "status": "ok",
            "engine": "light-lpca-turbo-v2",
            "groq_ready": bool(os.environ.get("GROQ_API_KEY")),
            "rate_limit_remaining": rate_limiter.remaining(ip),
            "vault": v.stats(),
        }

    @app.get("/api/history/{session_id}")
    async def history(session_id: str = "CORE_STREAM", limit: int = 40) -> List[Dict[str, Any]]:
        return v.get_history(session_id=session_id, limit=min(limit, 100))

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        ip = get_client_ip(request)
        if not rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a few seconds.")
        v.add_message("user", req.message, session_id=req.session_id)

        memories = v.search_memories(req.message, limit=6)
        mem_block = "\n".join(f"- [{m['category']}] {m['fact']}" for m in memories) or "None yet."

        research_block = ""
        if req.use_research:
            agent = TurboResearchAgent(vault=v, max_queries_per_dimension=1, max_pages_per_query=2)
            report = await agent.arun(req.message)
            research_block = (
                f"\nVerified Web Research (Confidence {report.confidence_score}%, {report.citation_count} sources):\n"
                + report.markdown_content[:2600]
            )

        sys_prompt = (
            "You are LIGHT-LPCA Turbo v2.0, a sovereign, ultra-fast cognitive assistant.\n"
            "If you learn a durable user preference, project detail, or key fact, include it inside "
            "<memory_update>...</memory_update> so the FTS5 vault saves it automatically.\n"
            f"Relevant Vault Memories:\n{mem_block}\n{research_block}"
        )
        history_msgs = v.get_history(session_id=req.session_id, limit=12)
        messages = [{"role": "system", "content": sys_prompt}] + [
            {"role": "assistant" if m["role"] == "ai" else m["role"], "content": m["content"]}
            for m in history_msgs
        ]

        async def event_gen():
            chunks: List[str] = []
            async for delta in router.stream_chat(
                messages,
                provider=req.provider,
                model=req.model,
                api_key=req.api_key,
            ):
                chunks.append(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            full_text = "".join(chunks)
            cleaned, saved_mems = extract_and_store_memories(full_text, v, source="chat")
            v.add_message("ai", cleaned or full_text, session_id=req.session_id, provider=req.provider or "auto")
            yield f"data: {json.dumps({'done': True, 'saved_memories': saved_mems})}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.post("/api/research")
    async def run_research(req: ResearchRequest) -> Dict[str, Any]:
        if not req.topic.strip():
            raise HTTPException(status_code=400, detail="Topic is required")
        agent = TurboResearchAgent(
            vault=v,
            max_queries_per_dimension=req.max_queries,
            max_pages_per_query=req.max_pages,
            use_cache=req.use_cache,
        )
        report = await agent.arun(req.topic)
        return agent.to_json(report)

    @app.get("/api/memory")
    async def get_memories(q: str = "", limit: int = 30) -> List[Dict[str, Any]]:
        return v.search_memories(q, limit=limit) if q.strip() else v.list_memories(limit=limit)

    @app.post("/api/memory")
    async def add_memory(req: MemoryAddRequest) -> Dict[str, Any]:
        mid = v.add_memory(req.fact, category=req.category, source="manual", importance=req.importance)
        return {"id": mid, "status": "stored"}

    @app.get("/api/shadow/status")
    async def shadow_status() -> Dict[str, Any]:
        return {
            "enabled": bool(v.get_setting("shadow_enabled", True)),
            "tasks": v.list_shadow_tasks(),
            "logs": v.get_shadow_logs(limit=25),
        }

    @app.post("/api/shadow/toggle")
    async def shadow_toggle(req: ShadowToggleRequest) -> Dict[str, Any]:
        v.set_setting("shadow_enabled", req.enabled)
        action = "SHADOW_PULSE_ENGAGED" if req.enabled else "SHADOW_PULSE_OFFLINE"
        details = "Autonomous background engine active" if req.enabled else "Neural silence enabled"
        v.log_shadow_event(action, details)
        return {"enabled": req.enabled, "action": action}

    @app.post("/api/shadow/tasks")
    async def add_shadow_task(req: ShadowTaskRequest) -> Dict[str, Any]:
        tid = v.add_shadow_task(req.title, req.query, interval_minutes=req.interval_minutes)
        v.log_shadow_event("SHADOW_TASK_QUEUED", f"Queued #{tid}: {req.title}")
        return {"id": tid, "status": "pending"}

    @app.post("/api/shadow/pulse")
    async def trigger_shadow_pulse() -> Dict[str, Any]:
        ran = await shadow.run_single_pulse()
        return {"executed": ran, "logs": v.get_shadow_logs(limit=10)}

    @app.get("/api/vault/providers")
    async def list_providers() -> List[Dict[str, Any]]:
        return router.list_providers_status()

    @app.post("/api/vault/providers")
    async def save_provider(req: ProviderSaveRequest) -> Dict[str, Any]:
        keys = v.get_setting("provider_keys", {})
        if req.api_key is not None and req.api_key.strip():
            keys[req.provider] = req.api_key.strip()
            v.set_setting("provider_keys", keys)
        if req.activate:
            v.set_setting(
                "active_llm",
                {
                    "provider": req.provider,
                    "model": req.model or "",
                    "api_key": keys.get(req.provider, ""),
                },
            )
        return {"status": "saved", "providers": router.list_providers_status()}

    @app.get("/api/vault/db_diagnostics")
    async def db_diagnostics() -> Dict[str, Any]:
        return v.db_diagnostics()

    @app.post("/api/vault/test_llm")
    async def test_llm(req: TestLLMRequest, request: Request) -> Dict[str, Any]:
        ip = get_client_ip(request)
        if not rate_limiter.allow(ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")
        import time as _time
        t0 = _time.perf_counter()
        chunks: List[str] = []
        async for d in router.stream_chat(
            [{"role": "user", "content": "Reply with exactly: Connection Established."}],
            provider=req.provider,
            model=req.model,
            api_key=req.api_key,
        ):
            chunks.append(d)
        reply = redact_secrets("".join(chunks).strip())
        ok = not reply.startswith("[") and "No external BYOK API key" not in reply
        return {
            "success": ok,
            "provider": req.provider,
            "latency_ms": round((_time.perf_counter() - t0) * 1000, 1),
            "response": reply[:240],
        }

    return app


app = create_app()
