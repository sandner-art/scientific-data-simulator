# simulator/ai/solver.py
"""A language model as a solver of bundle questions (text only).

:func:`solve_tasks` sends each task's prompt (with the public table, when the bundle publishes
one) and scores the reply with :func:`simulator.export.tasks.score`, which reads numbers,
choices and formulas out of free text. Figures are not sent: wrap a vision model yourself and
pass the image path from ``task["image"]`` if you need them.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence

from .clients import as_complete


def task_prompt(task: Mapping[str, Any], max_table_chars: int = 6000) -> str:
    prompt = str(task.get("prompt") or task.get("question") or "")
    table = task.get("table")
    if table and os.path.exists(str(table)):
        with open(str(table), encoding="utf-8", errors="replace") as f:
            text = f.read()
        if len(text) > max_table_chars:
            text = text[:max_table_chars] + "\n...(truncated)"
        prompt = f"{prompt}\n\nData table (CSV):\n{text}"
    return prompt


def solve_tasks(tasks: Sequence[Mapping[str, Any]], client: Any,
                responses: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    """``{task id: {"response", "score"}}`` for every task (answers must be in the tasks)."""
    from ..export.tasks import score
    complete = as_complete(client)
    out: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        reply = complete(task_prompt(task))
        if responses is not None:
            responses[task["id"]] = reply
        out[task["id"]] = {"response": reply, "score": float(score(task, reply))}
    return out
