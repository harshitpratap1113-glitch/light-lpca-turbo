"""
One-Command Launcher for LIGHT-LPCA Terminal Sleep Worker.
Usage after git clone:
    python sleep.py "Your task here"
    python sleep.py                       (interactive prompt)
    python sleep.py --resume <run_id>     (resume interrupted run)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from lpca_turbo.terminal_agent import SleepWorker, load_or_prompt_groq_key


def main() -> None:
    args = sys.argv[1:]
    resume_id = None
    if "--resume" in args:
        idx = args.index("--resume")
        if idx + 1 < len(args):
            resume_id = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]

    goal = " ".join(args).strip()
    if not goal and not resume_id:
        print("\n=== LIGHT-LPCA Terminal Sleep Worker ===")
        goal = input("Apna task likho (aur phir so jao): ").strip()

    if not goal and not resume_id:
        print("Error: Task empty nahi ho sakta.")
        sys.exit(1)

    api_key = load_or_prompt_groq_key(interactive=True)
    worker = SleepWorker(output_root=Path.cwd() / "lpca_sleep_output", api_key=api_key)
    asyncio.run(worker.arun(goal, resume_run_id=resume_id))


if __name__ == "__main__":
    main()
