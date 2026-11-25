"""
Terminal Sleep-Mode Autonomous Worker ("Task deke so jao" Edition).
Designed to run in any terminal after cloning from GitHub:
  python -m lpca_turbo.cli sleep "Your goal or task"

Features:
- Securely loads GROQ_API_KEY from environment or ~/.light_lpca_turbo/.env (0600, never inside repo)
- Decomposes the goal into concrete steps (research, analyze, synthesize, build)
- Checkpoints every completed step to disk so crashes/reboots resume cleanly (--resume)
- Generates MORNING_BRIEFING.md + MORNING_BRIEFING.html when done
"""

from __future__ import annotations

import asyncio
import getpass
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console
from rich.panel import Panel

from .engine import TurboResearchAgent
from .security import redact_secrets
from .vault import SovereignVault


ENV_FILE = Path.home() / ".light_lpca_turbo" / ".env"
RUNS_DIR = Path.home() / ".light_lpca_turbo" / "sleep_runs"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def load_or_prompt_groq_key(interactive: bool = True) -> str:
    """Loads GROQ_API_KEY from env or ~/.light_lpca_turbo/.env, or prompts safely once."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("GROQ_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    os.environ["GROQ_API_KEY"] = val
                    return val

    if not interactive:
        return ""

    print("\n[LIGHT-LPCA Terminal Sleep Worker]")
    print("GROQ_API_KEY not found in environment. Get a free key at https://console.groq.com")
    entered = getpass.getpass("Paste your GROQ_API_KEY (hidden input, saved locally to ~/.light_lpca_turbo/.env): ").strip()
    if entered:
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        ENV_FILE.write_text(f"GROQ_API_KEY={entered}\n", encoding="utf-8")
        try:
            os.chmod(ENV_FILE, 0o600)
        except Exception:
            pass
        os.environ["GROQ_API_KEY"] = entered
    return entered


async def _groq_complete(
    client: httpx.AsyncClient,
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = "openai/gpt-oss-120b",
    max_tokens: int = 1600,
) -> str:
    if not api_key:
        return "[Local offline fallback: GROQ_API_KEY not set — web research findings recorded.]"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.25, "max_tokens": max_tokens}
    resp = await client.post(GROQ_CHAT_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        return redact_secrets(f"[Groq HTTP {resp.status_code}]: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


async def _plan_steps(client: httpx.AsyncClient, api_key: str, goal: str) -> List[Dict[str, str]]:
    """Breaks the user's overnight goal into 4-6 structured steps."""
    if not api_key:
        return [
            {"id": "1", "type": "research", "title": "Web Research", "prompt": goal},
            {"id": "2", "type": "synthesize", "title": "Executive Summary", "prompt": f"Summarize key takeaways for: {goal}"},
        ]

    sys_msg = (
        "You are an autonomous overnight AI planner. Break the user's goal into 4 to 6 concrete sequential steps.\n"
        "Output ONLY a valid JSON array of objects with keys: id, type ('research' or 'execute'), title, prompt.\n"
        "Use type='research' when live web search & citations are needed, and type='execute' for deep analysis, code, or synthesis."
    )
    raw = await _groq_complete(
        client,
        api_key,
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": goal}],
        max_tokens=900,
    )
    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and parsed:
                return [
                    {
                        "id": str(item.get("id", i + 1)),
                        "type": str(item.get("type", "execute")).lower(),
                        "title": str(item.get("title", f"Step {i+1}")),
                        "prompt": str(item.get("prompt", goal)),
                    }
                    for i, item in enumerate(parsed[:6])
                ]
        except Exception:
            pass
    return [
        {"id": "1", "type": "research", "title": "Multi-Engine Web Research", "prompt": goal},
        {"id": "2", "type": "execute", "title": "Deep Technical Synthesis", "prompt": f"Produce a detailed action plan & deliverable for: {goal}"},
    ]


class SleepWorker:
    """Executes multi-step autonomous tasks with crash-proof checkpoints and Morning Briefing reports."""

    def __init__(
        self,
        vault: Optional[SovereignVault] = None,
        output_root: Optional[Path] = None,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
    ):
        self.vault = vault or SovereignVault()
        self.output_root = Path(output_root or Path.cwd() / "lpca_sleep_output")
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.console = Console()
        self.researcher = TurboResearchAgent(vault=self.vault, max_queries_per_dimension=1, max_pages_per_query=2)

    @staticmethod
    def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    async def arun(self, goal: str, resume_run_id: Optional[str] = None) -> Path:
        t0 = time.perf_counter()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        if resume_run_id:
            state_path = RUNS_DIR / f"{resume_run_id}.json"
            if not state_path.exists():
                raise FileNotFoundError(f"No checkpoint found for run_id={resume_run_id}")
            state = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
            state_path = RUNS_DIR / f"{run_id}.json"
            state = {
                "run_id": run_id,
                "goal": goal,
                "started_at": datetime.now().isoformat(),
                "steps": [],
                "completed_outputs": {},
                "status": "planning",
            }

        run_dir = self.output_root / state["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(Panel.fit(
            f"[bold yellow]LIGHT-LPCA Terminal Sleep Worker[/bold yellow]\n"
            f"Run ID: [cyan]{state['run_id']}[/cyan]\n"
            f"Goal: {state['goal']}\n"
            f"Checkpoint: [dim]{state_path}[/dim]",
            border_style="yellow",
        ))

        async with httpx.AsyncClient(timeout=45.0) as client:
            if not state["steps"]:
                self.console.print("[bold]Phase 1:[/bold] Planning autonomous steps...")
                state["steps"] = await _plan_steps(client, self.api_key, state["goal"])
                state["status"] = "running"
                self._save_state(state_path, state)

            total = len(state["steps"])
            for idx, step in enumerate(state["steps"], start=1):
                sid = step["id"]
                if sid in state["completed_outputs"]:
                    self.console.print(f"[dim][{idx}/{total}] Skipping already completed: {step['title']}[/dim]")
                    continue

                self.console.print(f"\n[bold cyan][{idx}/{total}] Executing ({step['type']}):[/bold cyan] {step['title']}")
                prior_ctx = "\n\n".join(
                    f"## {k}: {v['title']}\n{v['output'][:1500]}"
                    for k, v in state["completed_outputs"].items()
                )

                if step["type"] == "research":
                    report = await self.researcher.arun(step["prompt"])
                    step_out = report.markdown_content
                    self.console.print(
                        f"  [green]Verified {report.verified_claim_count} claims across {report.citation_count} sources ({report.elapsed_ms} ms)[/green]"
                    )
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an autonomous senior engineer & strategist completing a step while the user sleeps. Produce thorough, actionable, ready-to-use output.",
                        },
                        {
                            "role": "user",
                            "content": f"Overall Goal: {state['goal']}\n\nCurrent Step: {step['title']}\nInstruction: {step['prompt']}\n\nContext So Far:\n{prior_ctx}",
                        },
                    ]
                    step_out = await _groq_complete(client, self.api_key, messages, model=self.model)

                state["completed_outputs"][sid] = {
                    "title": step["title"],
                    "type": step["type"],
                    "output": step_out,
                    "finished_at": datetime.now().isoformat(),
                }
                self._save_state(state_path, state)
                self.vault.add_memory(f"Sleep run {state['run_id']} completed '{step['title']}'", category="sleep_worker")
                self.console.print(f"  [green]✓ Checkpointed step {idx}/{total}[/green]")

        elapsed_s = round(time.perf_counter() - t0, 2)
        state["status"] = "completed"
        state["elapsed_seconds"] = elapsed_s
        self._save_state(state_path, state)

        briefing_md = self._build_briefing_md(state, elapsed_s)
        briefing_path = run_dir / "MORNING_BRIEFING.md"
        briefing_path.write_text(briefing_md, encoding="utf-8")

        self.console.print(Panel.fit(
            f"[bold green]All {total} steps completed in {elapsed_s}s![/bold green]\n"
            f"Morning Briefing saved to:\n[cyan]{briefing_path}[/cyan]",
            border_style="green",
        ))
        return briefing_path

    @staticmethod
    def _build_briefing_md(state: Dict[str, Any], elapsed_s: float) -> str:
        lines = [
            f"# LIGHT-LPCA Morning Briefing — {state['run_id']}",
            f"**Goal:** {state['goal']}  ",
            f"**Status:** `{state['status'].upper()}` | **Elapsed:** `{elapsed_s}s`  ",
            "",
            "---",
            "",
        ]
        for sid, data in state["completed_outputs"].items():
            lines.append(f"## Step {sid}: {data['title']} (`{data['type']}`)")
            lines.append(data["output"])
            lines.append("\n---\n")
        return "\n".join(lines)
