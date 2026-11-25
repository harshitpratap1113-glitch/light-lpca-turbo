"""
Autonomous Proactive Shadow Brain Daemon.
Realizes LIGHT LPCA's core "Shadow Brain" vision:
- Runs continuously in the background when SHADOW_PULSE_ENGAGED is active
- Picks due tasks from SovereignVault, executes 100x TurboResearchAgent
- Stores distilled insights into FTS5 Memory & Shadow Logs
- Optionally notifies Telegram if TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID are set in environment
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx

from .engine import TurboResearchAgent
from .vault import SovereignVault


class ShadowBrain:
    """Proactive background cognitive loop for autonomous research & follow-up."""

    def __init__(self, vault: Optional[SovereignVault] = None, pulse_seconds: int = 30):
        self.vault = vault or SovereignVault()
        self.pulse_seconds = pulse_seconds
        self.agent = TurboResearchAgent(vault=self.vault, max_queries_per_dimension=1, max_pages_per_query=2)
        self._running = False

    async def _notify_telegram(self, text: str) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN") or self.vault.get_setting("telegram_bot_token", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID") or self.vault.get_setting("telegram_chat_id", "")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                await client.post(url, json={"chat_id": chat_id, "text": text[:3900]})
        except Exception:
            pass

    async def run_single_pulse(self) -> int:
        """Executes all currently due Shadow Tasks and logs outcomes. Returns number of tasks executed."""
        if not self.vault.get_setting("shadow_enabled", True):
            return 0
        due = self.vault.get_due_shadow_tasks()
        if not due:
            return 0

        executed = 0
        for task in due:
            report = await self.agent.arun(task["query"])
            summary = (
                f"[{task['title']}] Confidence={report.confidence_score}%, "
                f"Claims={report.verified_claim_count}, Citations={report.citation_count}, "
                f"Latency={report.elapsed_ms}ms"
            )
            self.vault.complete_shadow_task(task["id"], summary, report.confidence_score)
            self.vault.log_shadow_event("SHADOW_TASK_COMPLETED", summary)
            await self._notify_telegram(f"LIGHT-LPCA Shadow Pulse:\n{summary}")
            executed += 1
        return executed

    async def start_loop(self) -> None:
        self._running = True
        self.vault.log_shadow_event("SHADOW_PULSE_ENGAGED", "Autonomous background loop started.")
        while self._running:
            try:
                await self.run_single_pulse()
            except Exception as exc:
                self.vault.log_shadow_event("SHADOW_PULSE_ERROR", str(exc)[:200])
            await asyncio.sleep(self.pulse_seconds)

    def stop(self) -> None:
        self._running = False
        self.vault.log_shadow_event("SHADOW_PULSE_OFFLINE", "Autonomous background loop paused.")
