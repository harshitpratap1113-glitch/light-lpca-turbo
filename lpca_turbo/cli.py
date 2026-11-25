"""
Command Line Interface for LIGHT-LPCA Turbo v2.0.
Commands:
  lpca-turbo research "Topic" [--format md|html|json] [--out report.md]
  lpca-turbo serve [--port 8787]
  lpca-turbo shadow-pulse
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .engine import PHASE_LABELS, TOTAL_PHASES, TurboResearchAgent
from .shadow_brain import ShadowBrain
from .terminal_agent import SleepWorker, load_or_prompt_groq_key


def _score_style(score: float) -> str:
    if score >= 80:
        return "bold green"
    if score >= 50:
        return "bold yellow"
    return "bold red"


def build_summary(report, failed_queries) -> Panel:
    if not report.has_evidence:
        score_text = "[bold red]N/A[/bold red] [dim](no sources retrieved)[/dim]"
        style = "bold red"
    else:
        style = _score_style(report.confidence_score)
        score_text = f"[{style}]{report.confidence_score}%[/{style}]"

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="dim")
    table.add_column()
    table.add_row("Confidence Score", score_text)
    table.add_row("Latency", f"[cyan]{report.elapsed_ms:.1f} ms[/cyan] (cache hits: {report.cache_hits})")
    table.add_row("Verified Claims", str(report.verified_claim_count))
    table.add_row("Active Citations", str(report.citation_count))

    if failed_queries:
        dims = ", ".join(sorted({sq.dimension for sq in failed_queries}))
        table.add_row("Warning", f"[yellow]{len(failed_queries)} vector(s) returned no results ({dims})[/yellow]")

    title = "[bold]Turbo Research Complete[/bold]" if report.has_evidence else "[bold]Research Failed[/bold]"
    return Panel(table, title=title, border_style=style, expand=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="LIGHT-LPCA Turbo v2.0 CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_res = sub.add_parser("research", help="Run parallel multi-engine deep research")
    p_res.add_argument("topic", type=str, help="Research topic")
    p_res.add_argument("-o", "--out", type=str, help="Output file path")
    p_res.add_argument("--format", choices=["md", "html", "json"], default="md", help="Output format")
    p_res.add_argument("--queries", type=int, default=1, help="Queries per dimension")
    p_res.add_argument("--no-cache", action="store_true", help="Bypass SQLite cache")

    p_srv = sub.add_parser("serve", help="Start local-first Sovereign Workspace UI")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8787)

    sub.add_parser("shadow-pulse", help="Execute due Shadow Brain tasks once")

    p_slp = sub.add_parser("sleep", help="Run autonomous Terminal Sleep-Mode Worker on a task")
    p_slp.add_argument("goal", nargs="?", default="", help="Goal or task to execute autonomously")
    p_slp.add_argument("--resume", type=str, default=None, help="Resume an interrupted run_id")
    p_slp.add_argument("--out-dir", type=str, default=None, help="Directory to save MORNING_BRIEFING.md")

    args = parser.parse_args()
    console = Console(stderr=True)

    if args.cmd == "sleep":
        goal = args.goal.strip()
        if not goal and not args.resume:
            goal = input("Enter your task (then go to sleep): ").strip()
        if not goal and not args.resume:
            console.print("[red]Task cannot be empty.[/red]")
            return
        api_key = load_or_prompt_groq_key(interactive=True)
        worker = SleepWorker(output_root=Path(args.out_dir) if args.out_dir else None, api_key=api_key)
        asyncio.run(worker.arun(goal, resume_run_id=args.resume))
        return

    if args.cmd == "serve":
        import uvicorn
        console.print(f"[bold green]Starting LIGHT-LPCA Turbo v2.0[/bold green] at http://{args.host}:{args.port}")
        uvicorn.run("lpca_turbo.server:app", host=args.host, port=args.port, reload=False)
        return

    if args.cmd == "shadow-pulse":
        ran = asyncio.run(ShadowBrain().run_single_pulse())
        console.print(f"[bold green]Shadow Pulse executed {ran} task(s).[/bold green]")
        return

    if args.cmd != "research":
        parser.print_help()
        return

    agent = TurboResearchAgent(max_queries_per_dimension=args.queries, use_cache=not args.no_cache)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"[1/{TOTAL_PHASES}] Starting...", total=None)

        def on_progress(phase, detail="", advance=0, total=None):
            if total is not None:
                progress.update(task, total=total)
            desc = f"[{phase}/{TOTAL_PHASES}] {PHASE_LABELS[phase - 1]}"
            if detail:
                desc += f" {escape(detail)}"
            progress.update(task, description=desc, advance=advance)

        report = agent.run(args.topic, on_progress=on_progress)

    console.print(build_summary(report, agent.failed_queries))

    if args.format == "json":
        output = json.dumps(agent.to_json(report), indent=2)
    elif args.format == "html":
        output = report.html_content
    else:
        output = report.markdown_content

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Saved {args.format.upper()} report to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
