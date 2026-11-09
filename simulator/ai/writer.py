# simulator/ai/writer.py
"""A language model as a question writer: varied wording, checked against common leaks.

Template questions all read alike, which is itself a cue (and makes for narrow training data).
:func:`paraphrase_tasks` asks a model to rephrase each question and accepts the result only if
it passes checks that need no model:

* ``numbers``  - the rewrite has exactly the numbers of the original (none added, none lost),
  compared as values (``0.50`` = ``0.5``, ``1e-3`` = ``0.001``, ``10,000`` = ``10000``; a range
  ``0.5-8`` is two positive numbers) and in the same order; number words (``eight``, ``half``,
  ``percent``) may not be introduced;
* ``choices``  - every answer option named in the original is still named (as a whole word),
  in the original order, with no added emphasis (``**``, ``__``);
* ``answer``   - no answer from the dataset vocabulary (any choice answer, e.g. a law family; with
  ``_`` read as a space, a hyphen or nothing, and plural ``s``) appears unless it was in the
  original; for text and formula answers, the task's own answer may not appear;
* ``negation`` - no negation (``not``, ``never``, ``n't``) that the original did not have;
* ``length``   - between 0.5 and 3 times the original length, one line;
* ``prompt``   - the question occurs exactly once in the task prompt, so the prompt can be updated.

These checks catch the common leaks, not all: a synonym (``sigmoid`` for ``logistic``), a
changed unit or a subtle change of meaning passes them. Read a sample of accepted rewrites
before using a set, and keep the rejection report with it.

Rejected rewrites keep the original question; the report lists why. The answers never pass
through the model: they stay what the simulator computed. Each accepted task records the
writer model, so an evaluation can keep the writer and the test-taker apart (and the blind and
metadata baselines in :mod:`simulator.diagnostics.shortcuts` should be re-run on the rewritten
set: model wording brings its own cues).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .clients import as_complete, sha256_text

WRITER_INSTRUCTIONS = """Rewrite the question below so that it asks exactly the same thing in different words.
Rules: keep every number, unit, symbol and answer option exactly as written; do not add any number,
hint, explanation or new information; keep it one paragraph. Reply with the rewritten question only."""

# a sign belongs to the number only when nothing number-like precedes it ("0.5-8" is 0.5 and 8)
_NUMBER = re.compile(r"(?:(?<![\w.)\]])[-+])?(?<![A-Za-z_])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_NUMBER_WORDS = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
                           r"forty|fifty|hundred|thousand|million|half|third|quarter|double|twice|percent)\b", re.I)
_NEGATION = re.compile(r"\b(not|never|no|none|neither|nor)\b|n't\b", re.I)


def _normalise_numbers(text: str) -> str:
    text = text.replace("\u2212", "-")
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def _number_list(text: str) -> List[float]:
    out = []
    for m in _NUMBER.finditer(_normalise_numbers(text)):
        try:
            out.append(round(float(m.group()), 12))
        except ValueError:
            pass
    return out


def _numbers(text: str) -> Counter:
    return Counter(_number_list(text))


def _variants(term: str) -> List[str]:
    term = str(term).strip()
    forms = [term, term.replace("_", " "), term.replace("_", "-"), term.replace("_", "")]
    return list(dict.fromkeys(forms + [f + "s" for f in forms]))


def _contains(text: str, term: str) -> bool:
    low = text.lower()
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(v.lower()) + r"(?![A-Za-z0-9])", low)
               for v in _variants(term) if v)


def check_rewrite(original: str, rewrite: str, task: Mapping[str, Any], vocabulary: Sequence[str] = ()
                  ) -> List[str]:
    """Reasons to reject ``rewrite`` as a paraphrase of ``original`` (empty list: accepted)."""
    problems = []
    rewrite = rewrite.strip()
    if not rewrite:
        return ["empty"]
    if not 0.5 * len(original) <= len(rewrite) <= 3 * len(original) + 40 or "\n" in rewrite:
        problems.append("length")
    if _number_list(rewrite) != _number_list(original) or \
            len(_NUMBER_WORDS.findall(rewrite)) > len(_NUMBER_WORDS.findall(original)):
        problems.append("numbers")
    named = [str(c) for c in task.get("choices") or [] if _contains(original, str(c))]
    if named:
        def first_position(text, term):
            hits = [m.start() for v in _variants(term)
                    for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(v.lower()) + r"(?![A-Za-z0-9])", text.lower())]
            return min(hits) if hits else None
        positions = [first_position(rewrite, c) for c in named]
        original_order = sorted(named, key=lambda c: first_position(original, c))
        rewrite_order = sorted([c for c, p in zip(named, positions) if p is not None],
                               key=lambda c: first_position(rewrite, c))
        emphasis = any(mark in rewrite and mark not in original for mark in ("**", "__", "*", "!"))
        if any(p is None for p in positions) or rewrite_order != original_order or emphasis:
            problems.append("choices")
    if len(_NEGATION.findall(rewrite)) > len(_NEGATION.findall(original)):
        problems.append("negation")
    leaked = [t for t in vocabulary if _contains(rewrite, t) and not _contains(original, t)]
    answer = task.get("answer")
    if task.get("answer_type") in ("expression",) or (isinstance(answer, str) and task.get("answer_type") != "choice"):
        compact = re.sub(r"\s+", "", str(answer))
        if compact and compact in re.sub(r"\s+", "", rewrite) and compact not in re.sub(r"\s+", "", original):
            leaked.append(str(answer))
    if leaked:
        problems.append("answer")
    return problems


@dataclass
class RewriteReport:
    writer: str
    accepted: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)  # reason -> count
    examples: List[Dict[str, Any]] = field(default_factory=list)  # first rejected rewrites

    @property
    def total(self) -> int:
        return self.accepted + sum(self.rejected.values())


def answer_vocabulary(tasks: Sequence[Mapping[str, Any]]) -> List[str]:
    """Answers of choice questions across the set (e.g. all law families): words a rewrite must
    not introduce. Short tokens (fewer than 3 characters) are left out to avoid false alarms."""
    vocab = {str(t["answer"]) for t in tasks if t.get("answer_type") == "choice" and t.get("answer") is not None}
    return sorted(v for v in vocab if len(v) >= 3)


def paraphrase_tasks(tasks: Sequence[Mapping[str, Any]], client: Any, writer: str,
                     vocabulary: Optional[Sequence[str]] = None, instructions: str = WRITER_INSTRUCTIONS,
                     max_examples: int = 10) -> Tuple[List[Dict[str, Any]], RewriteReport]:
    """Rephrase the questions of ``tasks`` (which carry answers, privately) with a model.

    Returns ``(new_tasks, report)``. A new task has the rewritten ``question`` and ``prompt``
    (the question replaced inside the prompt) and a ``rewrite`` record (writer, original question
    hash); a rejected rewrite leaves the task unchanged. ``writer`` names the model.
    """
    complete = as_complete(client)
    vocab = list(vocabulary) if vocabulary is not None else answer_vocabulary(tasks)
    report = RewriteReport(writer=writer)
    out: List[Dict[str, Any]] = []
    for task in tasks:
        task = dict(task)
        original = str(task.get("question") or "")
        if not original:
            out.append(task)
            continue
        prompt = f"{instructions}\n\nQUESTION\n{original}"
        if task.get("choices"):
            prompt += f"\n\n(Answer options, keep them verbatim: {', '.join(map(str, task['choices']))})"
        rewrite = complete(prompt + "\n\nREWRITTEN QUESTION")
        rewrite = rewrite.strip().strip('"').strip()
        problems = check_rewrite(original, rewrite, task, vocab)
        if problems:
            for p in problems:
                report.rejected[p] = report.rejected.get(p, 0) + 1
            if len(report.examples) < max_examples:
                report.examples.append({"id": task.get("id"), "original": original, "rewrite": rewrite,
                                        "problems": problems})
            out.append(task)
            continue
        prompt_text = str(task.get("prompt") or "")
        if prompt_text and prompt_text.count(original) != 1:
            problems = ["prompt"]  # the prompt could not be updated consistently
        if problems:
            report.rejected["prompt"] = report.rejected.get("prompt", 0) + 1
            if len(report.examples) < max_examples:
                report.examples.append({"id": task.get("id"), "original": original, "rewrite": rewrite,
                                        "problems": problems})
            out.append(task)
            continue
        report.accepted += 1
        if prompt_text:
            task["prompt"] = prompt_text.replace(original, rewrite)
        task["question"] = rewrite
        task["rewrite"] = {"writer": writer, "original_sha256": sha256_text(original)}
        out.append(task)
    return out, report
