# simulator/ai/agent.py
"""A language model as the agent in an :class:`simulator.envs.ExperimentEnv` episode.

:class:`LLMAgent` is an agent for :func:`simulator.diagnostics.episodes.run_episodes`: each
turn it sends the task, the episode so far and the action format, and parses one JSON action
from the reply. Replies that do not parse get one correction round (``max_retries``); after
that the agent answers ``"<unparseable reply>"``, which the environment scores as wrong and
the diagnostics flag as ``invalid_answer``. When a reply holds several JSON objects (a worked
example, a draft in a thinking block, then the action), the *last* valid action is used. The prompt is rebuilt from the episode each turn
(no hidden chat state), so a :class:`~simulator.ai.clients.ReplayClient` reproduces episodes.

    from simulator.ai import LLMAgent, LoggedClient
    from simulator.diagnostics import run_episodes, diagnose_episode, law_alternatives, summarize

    client = LoggedClient(my_model, "logs/agent.jsonl", role="agent", model="provider/model-x")
    logs = run_episodes(env, LLMAgent(client), seeds=range(20))
    print(summarize([diagnose_episode(log, law_alternatives(log)) for log in logs]))
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional

from .clients import as_complete, json_candidates

INSTRUCTIONS = """You are a scientist running experiments on an instrument with a hidden property.
Each turn, reply with exactly one JSON object and nothing else:
  {"type": "experiment", "inputs": {<name>: <number>, ...}}   to run an experiment, or
  {"type": "answer", "value": <your answer>}                  to finish.
Experiments cost budget; the episode ends when you answer or the budget runs out.
Base your answer on the measurements you obtain."""

UNPARSEABLE = "<unparseable reply>"


def _render_step(i: int, step: Dict[str, Any], max_chars: int) -> str:
    obs = {k: v for k, v in step["observation"].items() if k != "done"}
    text = json.dumps(obs, separators=(",", ":"))
    if len(text) > max_chars:  # thin the measurement lists rather than cut the JSON mid-value
        thinned = json.loads(text)
        for entry in (thinned.get("measurements") or {}).values():
            for key, values in list(entry.items()):
                if isinstance(values, list) and len(values) > 8:
                    entry[key] = values[::int(math.ceil(len(text) / max_chars)) + 1]
        text = json.dumps(thinned, separators=(",", ":"))
        if len(text) > max_chars:
            text = json.dumps({"note": "measurements too long to show", "keys": sorted(obs)})
    return f"Turn {i + 1}. Your action: {json.dumps(step['action'])}\nResult: {text}"


def _validate(action: Any) -> Optional[str]:
    if not isinstance(action, dict):
        return "the reply must be a JSON object"
    kind = action.get("type")
    if kind == "experiment":
        if not isinstance(action.get("inputs"), dict):
            return "an experiment needs \"inputs\": {name: number, ...}"
        return None
    if kind == "answer":
        if "value" not in action:
            return "an answer needs \"value\""
        return None
    return "\"type\" must be \"experiment\" or \"answer\""


class LLMAgent:
    """Agent driven by a language model (any ``prompt -> text`` client).

    Args:
        client: the model, ideally a :class:`~simulator.ai.clients.LoggedClient` with role ``agent``.
        instructions: system-style preamble (the environment's task text follows it).
        max_retries: correction rounds for a reply that is not a valid JSON action.
        max_observation_chars: per-turn cap on the rendered measurements.
    """

    def __init__(self, client: Any, instructions: str = INSTRUCTIONS, max_retries: int = 1,
                 max_observation_chars: int = 4000):
        self._complete = as_complete(client)
        self.instructions = instructions
        self.max_retries = int(max_retries)
        self.max_observation_chars = int(max_observation_chars)
        self.stats = {"calls": 0, "parse_failures": 0, "gave_up": 0}

    def prompt(self, observation: Dict[str, Any], log) -> str:
        task = (log.task or {}).get("task") or observation.get("task", "")
        lines = [self.instructions, "", "TASK", task, ""]
        if log.steps:
            lines.append("EPISODE SO FAR")
            lines += [_render_step(i, s, self.max_observation_chars) for i, s in enumerate(log.steps)]
            lines.append("")
        budget = (log.task or {}).get("budget", observation.get("budget"))
        if budget is not None:
            used = sum(1 for s in log.steps if s["action"].get("type") == "experiment"
                       and "experiment" in s["observation"])  # failed experiments cost budget too
            left = max(0, int(budget) - used)
            lines.append(f"Experiments left: {left}." if left else
                         "No experiments left: you must answer now with {\"type\": \"answer\", \"value\": ...}.")
        lines.append("Your next action (one JSON object):")
        return "\n".join(lines)

    def __call__(self, observation: Dict[str, Any], log) -> Dict[str, Any]:
        prompt = self.prompt(observation, log)
        for attempt in range(self.max_retries + 1):
            self.stats["calls"] += 1
            reply = self._complete(prompt)
            candidates = json_candidates(reply)
            valid = [c for c in candidates if _validate(c) is None]
            if valid:
                action, problem = valid[-1], None
            else:
                action = None
                problem = _validate(candidates[-1]) if candidates else "no JSON object in the reply"
            if problem is None:
                return {"type": action["type"], **({"inputs": action["inputs"]} if action["type"] == "experiment"
                                                   else {"value": action["value"]})}
            self.stats["parse_failures"] += 1
            prompt = (f"{prompt}\n\nYour reply was:\n{reply[:2000]}\n\nIt could not be used ({problem}). "
                      "Reply with exactly one JSON object.")
        self.stats["gave_up"] += 1
        return {"type": "answer", "value": UNPARSEABLE}
