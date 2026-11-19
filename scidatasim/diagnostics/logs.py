"""Fallacy flags for any log: episodes recorded outside the simulator.

An LLM agent's tool calls, a human labelling session, a lab notebook exported as events: if a log
says, per episode, what was looked at or measured and what was finally answered, and an answer key
says what was true, the agent-fallacy flags of :mod:`scidatasim.diagnostics.episodes` apply
unchanged - answering without evidence, committing early while budget was left, repeating an
experiment, sampling too narrowly, contradicting one's own data, preferring a prior (an
alternative answer) to the evidence.

**Events** (JSONL, JSON list, CSV/TSV/Parquet; one row per event, in order within an episode):

``episode``     the episode (or item) id - required
``type``        ``experiment``/``measure``/``observe``/``query``/``tool``/``view`` (evidence) or
                ``answer``/``submit``/``label`` (an answer; the last one counts) - required
``inputs``      what was chosen (a mapping, or JSON text in tables); repeats are compared on it
``x``, ``y``    measured values (numbers or lists), for sampling and fit checks
``error``       non-empty when the step failed (it does not count as evidence)
``value``       the answer, for answer events
``step``        order within the episode (default: file order)

**Answer key** (JSON/YAML mapping or list, or a table) per episode: ``answer`` (required),
``answer_type`` (``number``, ``choice``, ``text``, ``expression``; inferred when missing),
``tolerance``, ``budget`` (experiments available), ``domain`` ([low, high] of x, for
``narrow_sampling`` and expression scoring) and ``alternatives`` (``{name: value}``: answers a prior
would give, e.g. the textbook value; flags ``matches_<name>``).

Command line: ``python -m scidatasim.diagnostics.logs EVENTS --key KEY [--json]`` or
``sds fallacies EVENTS --key KEY``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .episodes import EpisodeLog, diagnose_episode, summarize

EVIDENCE = {"experiment", "measure", "observe", "query", "tool", "view"}
ANSWER = {"answer", "submit", "label"}


def _records(path: str) -> List[Dict[str, Any]]:
    low = path.lower()
    if low.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if low.endswith((".json", ".yaml", ".yml")):
        with open(path, encoding="utf-8") as f:
            if low.endswith(".json"):
                data = json.load(f)
            else:
                import yaml
                data = yaml.safe_load(f)
        return data
    from ..ingest import read_table
    frame = read_table(path).frame
    return [{k: v for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))}
            for row in frame.to_dict("records")]


def _maybe_json(value):
    if isinstance(value, str) and value.strip()[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def read_key(source) -> Dict[str, Dict[str, Any]]:
    """``{episode: {"answer", "answer_type", "tolerance", "budget", "domain", "alternatives"}}``."""
    data = _records(source) if isinstance(source, (str, os.PathLike)) else source
    if isinstance(data, dict):
        rows = [dict(v, episode=k) if isinstance(v, dict) else {"episode": k, "answer": v} for k, v in data.items()]
    else:
        rows = list(data)
    key = {}
    for row in rows:
        row = {k: _maybe_json(v) for k, v in row.items()}
        if "episode" not in row or "answer" not in row:
            raise KeyError("every answer-key entry needs 'episode' and 'answer'")
        a = row["answer"]
        kind = row.get("answer_type") or ("number" if isinstance(a, (int, float)) and not isinstance(a, bool)
                                           else "text")
        key[str(row["episode"])] = {"answer": a, "answer_type": kind, "tolerance": float(row.get("tolerance") or 0.0),
                                    "budget": int(row["budget"]) if row.get("budget") not in (None, "") else None,
                                    "domain": row.get("domain"), "alternatives": row.get("alternatives") or {},
                                    "choices": row.get("choices")}
    return key


def _correct(spec: Dict[str, Any], value) -> Optional[bool]:
    if value is None:
        return None
    kind = spec["answer_type"]
    if kind == "expression":
        from ..expressions import ExpressionError, compare
        try:
            return compare(str(value), str(spec["answer"]), tuple(spec["domain"]))["relative_rms"] <= (
                spec["tolerance"] or 0.01)
        except (ExpressionError, TypeError, ValueError):
            return None
    if kind in ("number", "integer"):
        try:
            return abs(float(value) - float(spec["answer"])) <= max(spec["tolerance"], 1e-9 * max(1.0, abs(float(spec["answer"]))))
        except (TypeError, ValueError):
            return None
    return str(value).strip().lower() == str(spec["answer"]).strip().lower()


def episodes_from_events(events: Iterable[Dict[str, Any]], key: Dict[str, Dict[str, Any]]) -> List[EpisodeLog]:
    """Group events by episode and turn each into an :class:`EpisodeLog` the diagnostics read."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for i, e in enumerate(events):
        e = {k: _maybe_json(v) for k, v in e.items()}
        if "episode" not in e or "type" not in e:
            raise KeyError(f"event {i} needs 'episode' and 'type'")
        e.setdefault("_order", i)
        grouped.setdefault(str(e["episode"]), []).append(e)
    logs = []
    for episode in sorted(set(grouped) | set(key), key=str):
        spec = key.get(episode)
        if spec is None:
            continue  # no truth for this episode: nothing to diagnose
        rows = sorted(grouped.get(episode, []), key=lambda e: (e.get("step", e["_order"]), e["_order"]))
        steps, n = [], 0
        for e in rows:
            kind = str(e["type"]).lower()
            if kind in EVIDENCE:
                n += 1
                obs: Dict[str, Any] = {"experiment": n}
                if e.get("error") not in (None, "", False):
                    obs["error"] = str(e["error"])
                elif "x" in e or "y" in e:
                    x = np.atleast_1d(np.asarray(e.get("x", []), float)).tolist()
                    y = np.atleast_1d(np.asarray(e.get("y", []), float)).tolist()
                    obs["measurements"] = {"y": {"values": y, "x": x}}
                steps.append({"action": {"type": "experiment", "inputs": e.get("inputs", {})}, "observation": obs,
                              "reward": None, "info": {}})
            elif kind in ANSWER:
                value = e.get("value")
                ok = _correct(spec, value)
                steps.append({"action": {"type": "answer", "value": value}, "observation": {},
                              "reward": 1.0 if ok else 0.0, "info": {"correct": bool(ok), "estimate": value}})
        answers = [s for s in steps if s["action"]["type"] == "answer"]
        if len(answers) > 1:  # the last answer counts; earlier ones become notes
            for s in answers[:-1]:
                s["action"]["type"] = "revised_answer"
        log = EpisodeLog(seed=None, task={"episode": episode}, steps=steps, truth={"answer": spec["answer"]},
                         target="answer", target_type=spec["answer_type"], domain=spec["domain"],
                         tolerance=spec["tolerance"], budget=spec["budget"] if spec["budget"] is not None else 0)
        log.task["revisions"] = len(answers) - 1 if answers else 0
        log.task["budget_known"] = spec["budget"] is not None
        logs.append(log)
    return logs


def diagnose_log(events, key, misfit_threshold: float = 3.0) -> Dict[str, Any]:
    """Diagnose every episode of an event log against its answer key; returns ``{"episodes": [...],
    "summary": {...}}``. ``premature_commitment`` applies only where the key states a budget."""
    key = read_key(key) if isinstance(key, (str, os.PathLike)) else key
    events = _records(events) if isinstance(events, (str, os.PathLike)) else list(events)
    out = []
    for log in episodes_from_events(events, key):
        alternatives = {k: (str(v) if log.target_type == "expression" else v)
                        for k, v in (key[log.task["episode"]]["alternatives"] or {}).items()}
        d = diagnose_episode(log, alternatives, misfit_threshold)
        if not log.task["budget_known"]:
            d["flags"].pop("premature_commitment", None)
        if log.target_type != "expression":
            d["flags"].pop("narrow_sampling", None)
            d["flags"].pop("contradicts_own_data", None)
        d["episode"] = log.task["episode"]
        d["revisions"] = log.task["revisions"]
        out.append(d)
    return {"episodes": out, "summary": summarize(out)}


def markdown(result: Dict[str, Any]) -> str:
    s = result["summary"]
    lines = [f"# Fallacy flags: {s['episodes']} episodes", "",
             f"Accuracy {s['accuracy']:.3f}; mean evidence steps {s['mean_experiments']:.2f}.", "",
             "| flag | rate | count | applicable | accuracy when flagged |", "|---|---|---|---|---|"]
    for name, f in sorted(s["flags"].items()):
        acc = "-" if f["accuracy_when_flagged"] is None else f"{f['accuracy_when_flagged']:.2f}"
        lines.append(f"| {name} | {f['rate']:.3f} | {f['count']} | {f['applicable']} | {acc} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python -m scidatasim.diagnostics.logs", description=__doc__.split("\n")[0])
    parser.add_argument("events", help="event log: JSONL, JSON, CSV, TSV or Parquet")
    parser.add_argument("--key", required=True, help="answer key: JSON, YAML or a table")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = diagnose_log(args.events, args.key)
    print(json.dumps(result, indent=1, default=float) if args.json else markdown(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
