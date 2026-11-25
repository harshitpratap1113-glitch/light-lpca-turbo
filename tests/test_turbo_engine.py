"""Comprehensive test suite for LIGHT-LPCA Turbo v2.0."""

from __future__ import annotations

import asyncio
import time

from fastapi.testclient import TestClient

from lpca_turbo.auditor import FactAuditor
from lpca_turbo.byok_router import extract_and_store_memories
from lpca_turbo.engine import DeepResearchAgent, TurboResearchAgent
from lpca_turbo.planner import ResearchPlanner
from lpca_turbo.search_and_scrape import AsyncWebEngine, ScrapedPage
from lpca_turbo.security import SlidingWindowRateLimiter, redact_secrets
from lpca_turbo.server import create_app
from lpca_turbo.shadow_brain import ShadowBrain
from lpca_turbo.terminal_agent import SleepWorker
from lpca_turbo.vault import SovereignVault


SAMPLE_TEXT = "Artificial intelligence transforms enterprise software workflows at scale with 94% accuracy."


def _fake_page(url: str, dimension: str = "general") -> ScrapedPage:
    return ScrapedPage(
        url=url,
        title="Doc",
        content_markdown=SAMPLE_TEXT,
        status_code=200,
        dimension=dimension,
        word_count=len(SAMPLE_TEXT.split()),
        snippets=[SAMPLE_TEXT],
    )


def _stub_network(agent: TurboResearchAgent, monkeypatch, results):
    async def fake_search(client, query, max_results=3, engines=()):
        return list(results)

    async def fake_fetch(client, sem, url, dimension="general", use_cache=True):
        return _fake_page(url, dimension=dimension)

    monkeypatch.setattr(agent.scraper, "search_multi_engine", fake_search)
    monkeypatch.setattr(agent.scraper, "fetch_page", fake_fetch)


def test_research_planner_covers_all_dimensions():
    planner = ResearchPlanner(max_queries_per_dimension=2)
    sqs = planner.generate_plan("Autonomous AI Agents")
    dims = {sq.dimension for sq in sqs}
    assert dims == {"overview", "technical", "verification", "counter_perspective"}
    assert len(sqs) == 8


def test_html_cleaner_strips_scripts_and_nav():
    html = """
    <html><head><title>Turbo Test</title></head><body>
      <nav>Menu</nav><h1>Core Engine</h1>
      <p>Async concurrency reduces research latency across distributed nodes significantly.</p>
      <script>alert('bad')</script>
    </body></html>
    """
    title, md = AsyncWebEngine.clean_html_to_markdown(html)
    assert title == "Turbo Test"
    assert "Core Engine" in md
    assert "alert" not in md


def test_auditor_domain_diversity_and_contradiction():
    auditor = FactAuditor(min_verified_sources=2)
    pages = [
        ScrapedPage(
            url="https://alpha.org/a",
            title="A",
            content_markdown=SAMPLE_TEXT,
            status_code=200,
            dimension="overview",
            word_count=12,
            snippets=[SAMPLE_TEXT],
        ),
        ScrapedPage(
            url="https://beta.edu/b",
            title="B",
            content_markdown=SAMPLE_TEXT,
            status_code=200,
            dimension="verification",
            word_count=12,
            snippets=[SAMPLE_TEXT],
        ),
    ]
    res = auditor.audit_pages("AI", pages)
    assert res.has_evidence is True
    assert res.overall_confidence >= 80.0
    assert res.verified_claims[0].status == "VERIFIED"
    assert len(res.verified_claims[0].supporting_domains) == 2


def test_auditor_no_evidence_guard():
    res = FactAuditor(min_verified_sources=2).audit_pages("unreachable", [])
    assert res.has_evidence is False
    assert res.overall_confidence == 0.0
    assert res.citation_index == {}


def test_turbo_agent_and_cache_speed(tmp_path, monkeypatch):
    vault = SovereignVault(tmp_path / "vault.db")
    agent = TurboResearchAgent(vault=vault, max_queries_per_dimension=1, max_pages_per_query=2)
    _stub_network(
        agent,
        monkeypatch,
        [
            {"title": "A", "url": "https://domain-a.org/doc", "snippet": ""},
            {"title": "B", "url": "https://domain-b.org/doc", "snippet": ""},
        ],
    )

    report1 = agent.run("AI Hallucination Benchmarks")
    assert report1.has_evidence is True
    assert report1.confidence_score > 70.0
    assert "<html" in report1.html_content.lower()

    # Second run must hit report cache in a few milliseconds
    t0 = time.perf_counter()
    report2 = agent.run("AI Hallucination Benchmarks")
    cache_ms = (time.perf_counter() - t0) * 1000
    assert report2.topic == report1.topic
    assert cache_ms < 250.0


def test_progress_callback_exactness(tmp_path, monkeypatch):
    vault = SovereignVault(tmp_path / "vault.db")
    agent = DeepResearchAgent(vault=vault, max_queries_per_dimension=1, max_pages_per_query=2)
    _stub_network(agent, monkeypatch, [{"title": "A", "url": "https://domain-a.org/doc", "snippet": ""}])

    completed = 0
    total = None
    phases = []

    def on_progress(phase, detail="", advance=0, declared_total=None):
        nonlocal completed, total
        if declared_total is not None:
            total = declared_total
        completed += advance
        phases.append(phase)

    agent.run("progress topic", on_progress=on_progress)
    assert total is not None
    assert completed == total
    assert sorted(set(phases)) == [1, 2, 3, 4]


def test_vault_fts5_memory_and_tag_extraction(tmp_path):
    vault = SovereignVault(tmp_path / "vault.db")
    raw = "Sure, I got that!\n<memory_update>\n- User builds sovereign AI tools\n- Preferred model is Groq Llama 3.3\n</memory_update>"
    cleaned, saved = extract_and_store_memories(raw, vault)
    assert "<memory_update>" not in cleaned
    assert len(saved) == 2

    hits = vault.search_memories("sovereign")
    assert len(hits) == 1
    assert "sovereign" in hits[0]["fact"].lower()


def test_shadow_brain_pulse_executes_due_task(tmp_path, monkeypatch):
    vault = SovereignVault(tmp_path / "vault.db")
    vault.add_shadow_task("Track LLM Speed", "LLM inference latency 2026", interval_minutes=60)
    shadow = ShadowBrain(vault=vault)
    _stub_network(
        shadow.agent,
        monkeypatch,
        [
            {"title": "A", "url": "https://one.org/x", "snippet": ""},
            {"title": "B", "url": "https://two.org/y", "snippet": ""},
        ],
    )
    ran = asyncio.run(shadow.run_single_pulse())
    assert ran == 1
    tasks = vault.list_shadow_tasks()
    assert tasks[0]["status"] == "active"
    assert tasks[0]["last_run_at"] > 0


def test_fastapi_endpoints(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    vault = SovereignVault(tmp_path / "api_vault.db")
    app = create_app(vault=vault)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        h = client.get("/api/health").json()
        assert h["status"] == "ok"

        mem_res = client.post("/api/memory", json={"fact": "LIGHT LPCA Turbo uses SQLite WAL"})
        assert mem_res.status_code == 200
        found = client.get("/api/memory?q=SQLite").json()
        assert len(found) == 1

        tog = client.post("/api/shadow/toggle", json={"enabled": False}).json()
        assert tog["action"] == "SHADOW_PULSE_OFFLINE"

        provs = client.get("/api/vault/providers").json()
        assert any(p["id"] == "groq" for p in provs)

        diag = client.get("/api/vault/db_diagnostics").json()
        assert diag["status"] == "Connected"
        assert len(diag["collections"]) == 7

        test_res = client.post("/api/vault/test_llm", json={"provider": "groq", "api_key": ""}).json()
        assert test_res["success"] is False


def test_security_headers_rate_limit_and_redaction(tmp_path):
    assert redact_secrets("err gsk_abcdefghijklmnopqrstuvwxyz12345") == "err [REDACTED_KEY]"
    rl = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    assert rl.allow("1.2.3.4") is True
    assert rl.allow("1.2.3.4") is True
    assert rl.allow("1.2.3.4") is False

    vault = SovereignVault(tmp_path / "sec_vault.db")
    app = create_app(vault=vault)
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"
        assert "Content-Security-Policy" in r.headers


def test_sleep_worker_generates_morning_briefing(tmp_path, monkeypatch):
    vault = SovereignVault(tmp_path / "sleep.db")
    worker = SleepWorker(vault=vault, output_root=tmp_path / "out", api_key="")
    _stub_network(
        worker.researcher,
        monkeypatch,
        [
            {"title": "A", "url": "https://one.org/a", "snippet": ""},
            {"title": "B", "url": "https://two.org/b", "snippet": ""},
        ],
    )
    briefing = asyncio.run(worker.arun("Test Overnight Goal"))
    assert briefing.exists()
    content = briefing.read_text(encoding="utf-8")
    assert "MORNING BRIEFING" in content.upper()
    assert "Test Overnight Goal" in content
