# simulator/ai/clients.py
"""One way to call a language model, logged so the data stay reproducible.

A *client* is anything that turns a prompt into text: a function ``prompt -> text`` or an
:class:`simulator.base.LLMClient` (``generate_text``). The simulator has no provider dependency;
wrap the SDK you use in a function (see :mod:`simulator.llm`).

Model calls are not reproducible from a seed, so every call made for data generation or
evaluation goes through :class:`LoggedClient`, which appends one JSON line per call: role, model
name, settings, the prompt and output (or only their hashes) and the time. :class:`ReplayClient`
answers the same prompts from such a log, so a dataset or an evaluation can be regenerated
exactly without the model (or after the model is gone). :func:`llm_provenance` summarises logs
for a release without the prompt texts.

Roles used by :mod:`simulator.ai`: ``agent`` (plays environments), ``attacker`` (tries to recover
answers from a release without its evidence), ``writer`` (rephrases questions), ``solver``
(answers bundle questions). Recording the role and model of every call lets an evaluation keep
writer and test-taker apart.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Tuple, Union

ROLES = ("agent", "attacker", "writer", "solver", "author", "judge")


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def as_complete(client: Any) -> Callable[[str], str]:
    """``prompt -> text`` from a function, a :class:`LoggedClient` or an ``LLMClient``."""
    if hasattr(client, "generate_text") and callable(client.generate_text):
        return lambda prompt: str(client.generate_text(prompt))
    if callable(client):
        return lambda prompt: str(client(prompt))
    raise TypeError("client must be callable (prompt -> text) or have generate_text()")


class LoggedClient:
    """Wraps a client; appends every call to a JSON-lines log.

    Args:
        client: ``prompt -> text`` function or ``LLMClient``.
        log_path: JSONL file (created, appended to).
        role: what the calls are for (see :data:`ROLES`; other names are allowed).
        model: model name and version as you would cite it (e.g. ``"provider/model-2026-05"``).
        settings: sampling settings to record (temperature, max tokens, ...).
        store_text: keep prompt and output texts (needed by :class:`ReplayClient`); ``False``
            keeps only hashes and lengths.
    """

    def __init__(self, client: Any, log_path: str, role: str, model: str,
                 settings: Optional[Mapping[str, Any]] = None, store_text: bool = True):
        if not model:
            raise ValueError("name the model (and version) so the log can be cited")
        self._complete = as_complete(client)
        self.log_path = os.path.abspath(log_path)
        self.role = str(role)
        self.model = str(model)
        self.settings = dict(settings or {})
        self.store_text = bool(store_text)
        self.calls = 0
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    def with_role(self, role: str) -> "LoggedClient":
        """The same client and log under another role."""
        other = LoggedClient.__new__(LoggedClient)
        other.__dict__.update(self.__dict__)
        other.role = str(role)
        other.calls = 0
        return other

    def __call__(self, prompt: str) -> str:
        start = time.perf_counter()
        output = self._complete(prompt)
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "role": self.role,
                 "model": self.model, "settings": self.settings, "prompt_sha256": sha256_text(prompt),
                 "output_sha256": sha256_text(output), "prompt_chars": len(prompt), "output_chars": len(output),
                 "latency_s": round(time.perf_counter() - start, 3)}
        if self.store_text:
            entry["prompt"] = prompt
            entry["output"] = output
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.calls += 1
        return output


class ReplayMiss(KeyError):
    """The log holds no (more) answers for this prompt."""


class ReplayClient:
    """Answers prompts from a :class:`LoggedClient` log, in the order they were recorded.

    Identical prompts repeated in the log are answered in their recorded order. ``role``
    restricts the replay to one role. A prompt not in the log raises :class:`ReplayMiss` -
    a regeneration that asks something new is not a replay.
    """

    def __init__(self, log_path: str, role: Optional[str] = None):
        self.role = role
        self.model: Optional[str] = None
        self.skipped_last_line = False
        # keyed by prompt hash only: with role=None, answers come back in log order across roles
        self._answers: Dict[str, Deque[str]] = defaultdict(deque)
        with open(log_path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except ValueError:
                if i == len(lines) - 1:  # a call interrupted while writing: the rest is usable
                    self.skipped_last_line = True
                    continue
                raise
            if role is not None and entry.get("role") != role:
                continue
            if "output" not in entry:
                raise ValueError("the log was written with store_text=False and cannot be replayed")
            self.model = self.model or entry.get("model")
            self._answers[entry["prompt_sha256"]].append(entry["output"])

    def __call__(self, prompt: str) -> str:
        key_hash = sha256_text(prompt)
        queue = self._answers.get(key_hash)
        if queue:
            return queue.popleft()
        raise ReplayMiss(f"prompt {key_hash[:12]} is not (or no longer) in the log")


def llm_provenance(log_paths: Union[str, Iterable[str]]) -> Dict[str, Any]:
    """Summary of model use for a release: per role and model the number of calls and the time
    span, plus the SHA-256 of each log file (the logs themselves can stay private)."""
    if isinstance(log_paths, str):
        log_paths = [log_paths]
    usage: Dict[Tuple[str, str], Dict[str, Any]] = {}
    files: List[Dict[str, str]] = []
    for path in log_paths:
        with open(path, "rb") as f:
            files.append({"file": os.path.basename(path), "sha256": hashlib.sha256(f.read()).hexdigest()})
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                key = (e.get("role", "?"), e.get("model", "?"))
                u = usage.setdefault(key, {"role": key[0], "model": key[1], "calls": 0, "settings": e.get("settings"),
                                           "first": e.get("time"), "last": e.get("time")})
                u["calls"] += 1
                u["last"] = e.get("time")
    return {"llm_calls": sorted(usage.values(), key=lambda u: (u["role"], u["model"])), "logs": files}


def write_llm_provenance(directory: str, log_paths: Union[str, Iterable[str]]) -> str:
    """Write :func:`llm_provenance` to ``<directory>/llm_provenance.json``; returns the path."""
    path = os.path.join(directory, "llm_provenance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(llm_provenance(log_paths), f, indent=2)
    return path


def roles_of(log_paths: Union[str, Iterable[str]]) -> Dict[str, set]:
    """``{model: {roles}}`` - e.g. to check that the writer of a question set is not the model
    evaluated on it."""
    out: Dict[str, set] = defaultdict(set)
    for entry in llm_provenance(log_paths)["llm_calls"]:
        out[entry["model"]].add(entry["role"])
    return dict(out)


def json_candidates(text: str) -> List[Any]:
    """Every top-level JSON object or array in a model reply, in order of appearance (fenced
    blocks and prose around them allowed; nested values are not listed separately)."""
    decoder = json.JSONDecoder()
    found: List[Any] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] in "{[":
            try:
                value, end = decoder.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            found.append(value)
            i = end
        else:
            i += 1
    return found


def extract_json(text: str) -> Any:
    """The first JSON object or array in a model reply (code fences and prose around it allowed)."""
    found = json_candidates(text)
    if not found:
        raise ValueError("no JSON object in the reply")
    return found[0]
