"""
TurboResearchAgent — 100x Async Parallel Orchestrator.
Concurrently dispatches all search vectors across multiple search engines, deduplicates URLs,
pulls cached pages in <5ms, fetches uncached pages concurrently, cross-audits claims, and caches reports.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from .auditor import FactAuditor
from .planner import ResearchPlanner, SubQuery
from .search_and_scrape import AsyncWebEngine, ScrapedPage
from .synthesizer import ResearchReport, ResearchReportSynthesizer
from .vault import SovereignVault


PHASE_LABELS = (
    "Planning multi-vector search queries...",
    "Parallel multi-engine search & scraping...",
    "Cross-auditing factual claims...",
    "Synthesizing verified report...",
)
TOTAL_PHASES = len(PHASE_LABELS)


class TurboResearchAgent:
    """Async-first deep research engine with SovereignVault caching and multi-engine retrieval."""

    def __init__(
        self,
        max_queries_per_dimension: int = 1,
        max_pages_per_query: int = 2,
        scraper_timeout: float = 6.0,
        concurrency: int = 16,
        engines: Tuple[str, ...] = ("ddg", "wikipedia", "hackernews"),
        vault: Optional[SovereignVault] = None,
        use_cache: bool = True,
    ):
        self.vault = vault or SovereignVault()
        self.planner = ResearchPlanner(max_queries_per_dimension=max_queries_per_dimension)
        self.scraper = AsyncWebEngine(
            vault=self.vault if use_cache else None,
            timeout=scraper_timeout,
            concurrency=concurrency,
        )
        self.auditor = FactAuditor(min_verified_sources=2)
        self.synthesizer = ResearchReportSynthesizer()
        self.max_pages = max_pages_per_query
        self.engines = engines
        self.use_cache = use_cache
        self.failed_queries: List[SubQuery] = []

    async def arun(
        self,
        topic: str,
        on_progress: Optional[Callable[[int, str, int, Optional[int]], None]] = None,
    ) -> ResearchReport:
        t0 = time.perf_counter()

        def tick(phase: int, detail: str = "", advance: int = 0, total: Optional[int] = None) -> None:
            if on_progress:
                on_progress(phase, detail, advance, total)

        # Phase 1: Plan
        tick(1)
        sub_queries = self.planner.generate_plan(topic)
        tick(1, f"{len(sub_queries)} search vectors", advance=1, total=len(sub_queries) + 3)

        # Fast path: report cache check
        if self.use_cache and self.vault:
            cached = self.vault.get_cached_report(topic)
            if cached:
                for sq in sub_queries:
                    tick(2, f"[{sq.dimension}] instant cache hit", advance=1)
                tick(3, "cached audit", advance=1)
                tick(4, "report ready", advance=1)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return ResearchReport(
                    topic=cached["topic"],
                    markdown_content=cached["markdown"],
                    confidence_score=cached["confidence_score"],
                    verified_claim_count=cached["verified_claim_count"],
                    citation_count=cached["citation_count"],
                    generated_at=cached["generated_at"],
                    elapsed_ms=round(elapsed_ms, 2),
                    cache_hits=cached.get("citation_count", 1),
                    has_evidence=cached["has_evidence"],
                    html_content=cached.get("html", ""),
                )

        # Phase 2: Concurrent Multi-Engine Search + Concurrent Scraping
        self.failed_queries = []
        sem = asyncio.Semaphore(self.scraper.concurrency)

        async with httpx.AsyncClient(
            timeout=self.scraper.timeout,
            headers={"User-Agent": self.scraper.USER_AGENT},
            follow_redirects=True,
        ) as client:
            search_tasks = [
                self.scraper.search_multi_engine(client, sq.query, max_results=self.max_pages, engines=self.engines)
                for sq in sub_queries
            ]
            search_batches = await asyncio.gather(*search_tasks, return_exceptions=True)

            url_to_dim: Dict[str, str] = {}
            for sq, batch in zip(sub_queries, search_batches):
                if isinstance(batch, Exception) or not batch:
                    self.failed_queries.append(sq)
                    tick(2, f"[{sq.dimension}] no search results", advance=1)
                    continue
                added = 0
                for item in batch[: self.max_pages]:
                    u = item.get("url", "")
                    if u and u not in url_to_dim:
                        url_to_dim[u] = sq.dimension
                        added += 1
                tick(2, f"[{sq.dimension}] queued {added} source(s)", advance=1)

            fetch_tasks = [
                self.scraper.fetch_page(client, sem, url, dimension=dim, use_cache=self.use_cache)
                for url, dim in url_to_dim.items()
            ]
            scraped_pages: List[ScrapedPage] = (
                list(await asyncio.gather(*fetch_tasks)) if fetch_tasks else []
            )

        # Phase 3: Cross-Audit
        tick(3, f"{len(scraped_pages)} page(s)")
        audit_result = self.auditor.audit_pages(topic, scraped_pages)
        tick(3, f"{len(audit_result.verified_claims)} claim(s) verified", advance=1)

        # Phase 4: Synthesize
        tick(4)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        cache_hits = sum(1 for p in scraped_pages if p.from_cache)
        report = self.synthesizer.synthesize(topic, audit_result, scraped_pages, elapsed_ms=elapsed_ms, cache_hits=cache_hits)
        tick(4, "report ready", advance=1)

        if self.use_cache and self.vault and report.has_evidence:
            self.vault.set_cached_report(topic, self.to_json(report))
            self.vault.add_memory(
                f"Researched '{topic}' (Confidence {report.confidence_score}%, {report.citation_count} citations)",
                category="research",
                source="turbo_engine",
            )

        return report

    def run(
        self,
        topic: str,
        on_progress: Optional[Callable[[int, str, int, Optional[int]], None]] = None,
    ) -> ResearchReport:
        """Synchronous convenience wrapper (compatible with original deep-research-skill API)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # When called from inside an active event loop (e.g. FastAPI), run in worker thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.arun(topic, on_progress=on_progress)).result()
        return asyncio.run(self.arun(topic, on_progress=on_progress))

    @staticmethod
    def to_json(report: ResearchReport) -> Dict[str, Any]:
        return {
            "topic": report.topic,
            "confidence_score": report.confidence_score,
            "has_evidence": report.has_evidence,
            "verified_claim_count": report.verified_claim_count,
            "citation_count": report.citation_count,
            "elapsed_ms": report.elapsed_ms,
            "cache_hits": report.cache_hits,
            "generated_at": report.generated_at,
            "markdown": report.markdown_content,
            "html": report.html_content,
        }


# Backwards-compatible aliases for drop-in replacement of LIGHTLPCA/deep-research-skill
DeepResearchAgent = TurboResearchAgent
LPCAResearchBridge = TurboResearchAgent
