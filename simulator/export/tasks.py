# simulator/export/tasks.py
"""Bundle questions as self-contained, automatically scored tasks.

A task is one question about one run, with everything a system under test may
see (caption, question, figure/table paths, answer format) and, separately,
what a grader needs (answer, tolerance)::

    {"id": "<experiment_id>/q3", "prompt": "...", "question": "...",
     "image": "runs/<run>/bundle/figure.png", "table": "runs/<run>/bundle/table.csv",
     "answer_type": "number", "tolerance_rule": "3 % of the series' range", "choices": null,
     "answerable_from": "figure", "template": "max", "experiment_id": "...", "seed": ...,
     "model": "...", "fidelity": "reference",
     "answer": 13.68, "tolerance": 0.39}    # only in the private file

Score responses with :func:`score` (same rules as :func:`simulator.bundle.check_answer`).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from ..bundle import PRIVATE_QUESTION_KEYS, check_answer

PRIVATE_KEYS = PRIVATE_QUESTION_KEYS  # answer and the (answer-derived) numeric tolerance

_FORMAT = {
    "number": "Respond with a single number.",
    "integer": "Respond with a single integer.",
}


def _answer_format(item: Dict[str, Any]) -> str:
    if item["answer_type"] == "choice":
        return "Respond with exactly one of: " + ", ".join(item.get("choices") or []) + "."
    return _FORMAT[item["answer_type"]]


def _run_dirs(paths: Union[str, Sequence[str]]) -> List[str]:
    """Run directories under ``paths`` (a run dir, a design dir, or a list of either)."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    found: List[str] = []
    for path in paths:
        path = os.fspath(path)
        if os.path.exists(os.path.join(path, "experiment_record.json")):
            found.append(path)
            continue
        for name in sorted(os.listdir(path)):
            candidate = os.path.join(path, name)
            if os.path.exists(os.path.join(candidate, "experiment_record.json")):
                found.append(candidate)
    return found


def collect_bundle_tasks(paths: Union[str, Sequence[str]], relative_to: Optional[str] = None,
                         include_caption: bool = True) -> List[Dict[str, Any]]:
    """Collect tasks from the bundles of runs under ``paths``.

    Args:
        paths: A run directory, a design directory, or a list of either.
        relative_to: Make ``image``/``table`` paths relative to this directory
            (default: absolute paths).
        include_caption: Put the figure caption into the prompt.
    """
    tasks: List[Dict[str, Any]] = []
    for run_dir in _run_dirs(paths):
        bundle_dir = os.path.join(run_dir, "bundle")
        manifest_path = os.path.join(bundle_dir, "bundle.json")
        qa_path = os.path.join(bundle_dir, "qa.json")
        if not (os.path.exists(manifest_path) and os.path.exists(qa_path)):
            continue
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        with open(qa_path, encoding="utf-8") as f:
            items = json.load(f)

        def path_of(name: Optional[str]) -> Optional[str]:
            if not name:
                return None
            full = os.path.abspath(os.path.join(bundle_dir, name))
            return os.path.relpath(full, relative_to) if relative_to else full

        figure = (manifest.get("figure") or {}).get("output")
        table_entry = manifest.get("files", {}).get("table.csv") or {}
        has_table = table_entry.get("visibility") == "public"
        for item in items:
            parts = []
            if include_caption and manifest.get("caption"):
                parts.append(manifest["caption"])
            if figure and item.get("answerable_from") == "figure":
                parts.append("Use the attached figure.")
            parts.append(f"Question: {item['question']}")
            parts.append(_answer_format(item))
            tasks.append({
                "id": f"{manifest['experiment_id']}/{item['id']}",
                "prompt": "\n\n".join(parts),
                "question": item["question"],
                "image": path_of(figure),
                "table": path_of("table.csv") if has_table else None,
                "answer_type": item["answer_type"],
                "tolerance": item.get("tolerance"),
                "tolerance_rule": item.get("tolerance_rule"),
                "choices": item.get("choices"),
                "answerable_from": item.get("answerable_from"),
                "template": item.get("template"),
                "experiment_id": manifest["experiment_id"],
                "seed": manifest.get("seed"),
                "model": manifest.get("model"),
                "fidelity": manifest.get("fidelity"),
                "answer": item["answer"],
            })
    return tasks


def public_view(task: Dict[str, Any], include_seeds: bool = False) -> Dict[str, Any]:
    """What a system under test may see: no answer, no numeric tolerance and, by
    default, no seed (a seed plus the simulator regenerates the run, answers included)."""
    hidden = set(PRIVATE_KEYS) | (set() if include_seeds else {"seed"})
    return {k: v for k, v in task.items() if k not in hidden}


def write_tasks_jsonl(tasks: Iterable[Dict[str, Any]], path: str, include_answers: bool = False,
                      include_seeds: bool = False) -> int:
    """Write tasks as JSON Lines; answers (and seeds) only if asked. Returns the count."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task if include_answers else public_view(task, include_seeds)) + "\n")
            n += 1
    return n


def read_tasks_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_SUPERSCRIPTS = str.maketrans("", "", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")


def extract_number(text: str) -> Optional[str]:
    """The answer number in free text.

    Uses the content of ``\\boxed{...}`` if present and an explicit
    "answer is/: X" if present, otherwise the last number; ignores unit
    exponents (``cm^2``, ``m**3``, ``m\u00b2``) and thousands separators (``1,234.5``).
    """
    import re
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if boxed:
        text = boxed[-1]
    text = text.translate(_SUPERSCRIPTS)
    text = re.sub(r"(?:\^|\*\*)\s*\{?\s*[-+]?\d+\s*\}?", "", text)  # unit exponents
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)               # thousands separators
    explicit = re.findall(r"answer\s*(?:is|:|=)\s*(" + _NUMBER + ")", text, re.I)
    if explicit:
        return explicit[-1]
    numbers = re.findall(_NUMBER, text)
    return numbers[-1] if numbers else None


def score(task: Dict[str, Any], response: Any) -> float:
    """1.0 if ``response`` answers ``task`` correctly, else 0.0. Free text is
    accepted (see :func:`extract_number`; for choices, the one choice named)."""
    if task["answer_type"] in ("number", "integer") and isinstance(response, str):
        response = extract_number(response)
        if response is None:
            return 0.0
    elif task["answer_type"] == "choice" and isinstance(response, str):
        lowered = response.strip().lower()
        matches = [c for c in task.get("choices") or [] if c.lower() in lowered]
        if len(matches) == 1:
            response = matches[0]
    return 1.0 if check_answer(task, response) else 0.0
