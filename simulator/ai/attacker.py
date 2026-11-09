# simulator/ai/attacker.py
"""A language model as an attacker: can it answer the questions from a release *without* the
evidence?

:func:`simulator.audit.audit_export` runs fixed attacks for known leak channels. A model can try
anything: read the dataset card, the Croissant metadata, the captions, the order of records, its
own memory of similar data. :func:`llm_attack` gives the model the release with the evidence
removed (figures, measurement arrays, data tables) and asks it to answer every question. Its
accuracy is compared per template with the blind baseline (answers from the other answers
only) and with chance; accuracy significantly above both means the release, or the model's
prior, answers questions that should need the data.

A FLAGGED result is a lead, not a verdict: read the model's answers and find the channel. A
model with strong priors (a typical value for a physical constant, say), or a caption that
states the plotted range, can beat the blind baseline legitimately; the counterfactual laws
exist to remove the first.
"""
from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .clients import as_complete, json_candidates

EVIDENCE_PATTERNS = ("*.png", "*.svg", "*.pdf", "*.jpg", "*.jpeg", "*.npy", "*.npz", "*.h5", "*.hdf5",
                     "table.csv", "elements.csv", "plot_figure.py")
#: Never shown: answer keys and private records, wherever they turn up.
PRIVATE_NAMES = ("qa.json", "qa.jsonl", "description.md", "ground_truth.json", "experiment_record.json",
                 "design.csv", "design_results.csv", "design_manifest.json", "bundle.json")
#: Directories whose files are evidence (the Hugging Face export's public tables).
EVIDENCE_DIRS = ("tables", "images")
#: Parquet columns left out of the view because every question is listed with them anyway.
LISTED_COLUMNS = ("prompt", "question", "choices", "answer_type")
TEXT_SUFFIXES = (".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".csv", ".dot", ".py")

ATTACK_INSTRUCTIONS = """You are auditing a dataset release for answer leakage.
The figures and measurement data have been REMOVED from the release; only the files below remain.
Answer every question as well as you can from these files alone (and anything you already know).
If you cannot infer an answer, give your best guess anyway - never skip a question.
Reply with one JSON object mapping each question id to its answer, and nothing else."""


def _is_evidence(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _bundle_visibility(directory: str) -> Optional[Dict[str, str]]:
    path = os.path.join(directory, "bundle.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            files = json.load(f).get("files", {})
        return {name: info.get("visibility", "private") for name, info in files.items()}
    except (OSError, ValueError, AttributeError):
        return {}


@dataclass
class ReleaseView:
    """The evidence-free text of a release, and what was left out or cut."""
    text: str
    included: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)
    truncated: List[str] = field(default_factory=list)


def build_view(public_dir: str, max_chars: int = 150000, max_file_chars: int = 60000,
               evidence_patterns: Sequence[str] = EVIDENCE_PATTERNS) -> ReleaseView:
    """See :func:`evidence_free_view`; also lists included, excluded and truncated files."""
    import pandas as pd
    parts: List[str] = []
    view = ReleaseView(text="")
    for root, dirs, files in os.walk(public_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        visibility = _bundle_visibility(root)
        in_evidence_dir = any(part in EVIDENCE_DIRS for part in os.path.relpath(root, public_dir).split(os.sep))
        for name in sorted(files):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, public_dir).replace(os.sep, "/")  # the same text on every OS
            private = name in PRIVATE_NAMES or (visibility is not None and visibility.get(name) != "public")
            if private or in_evidence_dir or _is_evidence(name, evidence_patterns):
                view.excluded.append(rel)
                continue
            text = None
            if name.endswith(".parquet"):
                try:
                    frame = pd.read_parquet(path)
                except Exception:
                    view.excluded.append(rel)
                    continue
                keep = [c for c in frame.columns if c not in LISTED_COLUMNS and not frame[c].map(
                    lambda v: isinstance(v, (list, tuple, dict, bytes)) or hasattr(v, "__len__")
                    and not isinstance(v, str)).any()]
                text = frame[keep].to_csv(index=False)
            elif name.endswith(TEXT_SUFFIXES):
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            if text is None:
                view.excluded.append(rel)
                continue
            if len(text) > max_file_chars:
                text = text[:max_file_chars] + "\n...(truncated)"
                view.truncated.append(rel)
            parts.append(f"=== {rel} ===\n{text}")
            view.included.append(rel)
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(release view truncated)"
        view.truncated.append("(whole view)")
    view.text = text
    return view


def evidence_free_view(public_dir: str, max_chars: int = 150000, max_file_chars: int = 60000,
                       evidence_patterns: Sequence[str] = EVIDENCE_PATTERNS) -> str:
    """The release as text, minus the evidence and anything private.

    Left out: images, arrays and HDF5 files, data tables (``table.csv``, ``elements.csv``, the
    export's ``tables/``), figure scripts, list-valued (measurement) and image columns of Parquet
    files (and the question text columns, which the attack lists anyway), answer keys and private
    records (``qa.json``, ``description.md``, ...), and in a bundle
    folder every file its ``bundle.json`` does not mark public. Files are capped at
    ``max_file_chars`` and the view at ``max_chars``; :func:`build_view` reports any cut, and
    :func:`llm_attack` treats a cut view as inconclusive (a leak may sit in the cut part)."""
    return build_view(public_dir, max_chars, max_file_chars, evidence_patterns).text


@dataclass
class AttackReport:
    """Attacker accuracy against the blind baseline, per template.

    ``verdict``: ``PASS`` (no template answered above the null without the evidence),
    ``FLAGGED`` (some were: a leak, or a legitimate route such as a caption stating the range or a
    strong prior - inspect the answers), or ``INCONCLUSIVE`` (too few answers, no questions, or a
    truncated view)."""
    templates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_questions: int = 0
    n_answered: int = 0
    alpha: float = 0.05
    min_answered: float = 0.9
    truncated: List[str] = field(default_factory=list)
    answers: Dict[str, Any] = field(default_factory=dict)

    @property
    def leaks(self) -> List[str]:
        return [t for t, r in self.templates.items() if r["significant"]]

    @property
    def verdict(self) -> str:
        if self.n_questions == 0 or self.n_answered < self.min_answered * self.n_questions or self.truncated:
            return "INCONCLUSIVE"
        return "FLAGGED" if self.leaks else "PASS"

    @property
    def ok(self) -> bool:
        return self.verdict == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "leaks": self.leaks, "verdict": self.verdict, "ok": self.ok}

    def summary(self) -> str:
        lines = [f"LLM attack: {self.verdict} ({self.n_answered}/{self.n_questions} answered; "
                 f"Bonferroni alpha {self.alpha} over {len(self.templates)} templates)"]
        if self.truncated:
            lines.append(f"  view truncated: {', '.join(self.truncated)} - raise max_chars / max_file_chars")
        for t, r in sorted(self.templates.items()):
            mark = "  <-- answerable without the evidence: inspect" if r["significant"] else ""
            lines.append(f"  {t:<16} attacker {r['accuracy']:.2f}  blind {r['blind_accuracy']:.2f}  "
                         f"null {r['null_rate']:.2f}  p {r['p_value']:.4f}  (n {r['n']}){mark}")
        return "\n".join(lines)


def _answers_from_reply(reply: str, ids: Sequence[str]) -> Dict[str, Any]:
    """The candidate in a reply that answers most of ``ids``: a mapping ``id -> answer`` or a list
    of ``{"id", "answer"}`` records."""
    wanted = set(ids)
    best: Dict[str, Any] = {}
    for candidate in json_candidates(reply):
        found: Dict[str, Any] = {}
        if isinstance(candidate, dict):
            found = {k: v for k, v in candidate.items() if k in wanted}
        elif isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, dict) and item.get("id") in wanted:
                    found[item["id"]] = item.get("answer", item.get("value"))
        if len(found) > len(best):
            best = found
    return best


def llm_attack(client: Any, public_dir: str, private, batch_size: int = 20, max_chars: int = 150000,
               max_file_chars: int = 60000, alpha: float = 0.05, min_answered: float = 0.9,
               evidence_patterns: Sequence[str] = EVIDENCE_PATTERNS) -> AttackReport:
    """Ask a model to answer all questions of a release from its evidence-free view.

    Args:
        client: the attacking model (ideally a ``LoggedClient`` with role ``attacker``).
        public_dir: the release folder.
        private: run or design directory (or list) holding the answers.
        batch_size: questions per call (the release view is sent with every batch).
        min_answered: below this fraction of answered questions the verdict is INCONCLUSIVE.

    For each template the null rate is the best of: the blind baseline (majority answer or
    majority position, leave-one-out), chance (choices) and a floor of 0.5/n; the attacker's hits
    are tested against it (exact binomial, Bonferroni over templates).
    """
    from ..diagnostics.shortcuts import blind_baseline
    from ..diagnostics.stats import binomial_p_greater
    from ..export.tasks import collect_bundle_tasks, public_view, score
    complete = as_complete(client)
    tasks = collect_bundle_tasks(private)
    view = build_view(public_dir, max_chars=max_chars, max_file_chars=max_file_chars,
                      evidence_patterns=evidence_patterns)
    answers: Dict[str, Any] = {}
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        listing = []
        for t in batch:
            q = public_view(t)
            item = {"id": q["id"], "question": q.get("prompt") or q.get("question"), "answer_type": q["answer_type"]}
            if q.get("choices"):
                item["choices"] = q["choices"]
            listing.append(item)
        prompt = (f"{ATTACK_INSTRUCTIONS}\n\nRELEASE FILES (evidence removed)\n{view.text}\n\nQUESTIONS\n"
                  f"{json.dumps(listing, indent=1)}\n\nYour JSON object (id -> answer):")
        answers.update(_answers_from_reply(complete(prompt), [t["id"] for t in batch]))
    report = AttackReport(n_questions=len(tasks), n_answered=len(answers), alpha=alpha, min_answered=min_answered,
                          truncated=list(view.truncated), answers=answers)
    if not tasks:
        return report
    blind = blind_baseline(tasks)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        groups.setdefault(str(t.get("template")), []).append(t)
    n_templates = max(1, len(groups))
    for template, group in groups.items():
        hits = sum(score(t, answers[t["id"]]) >= 1.0 for t in group if t["id"] in answers)
        n = len(group)
        b = blind[template]
        null_rate = max(b["accuracy"], b.get("position_accuracy", 0.0), b.get("chance", 0.0), 0.5 / n)
        p = binomial_p_greater(int(hits), n, null_rate)
        p_adj = min(1.0, p * n_templates)
        report.templates[template] = {"n": n, "hits": int(hits), "accuracy": hits / n,
                                      "blind_accuracy": max(b["accuracy"], b.get("position_accuracy", 0.0)),
                                      "chance": b.get("chance"), "null_rate": null_rate, "p_value": p_adj,
                                      "significant": bool(hits / n > null_rate and p_adj < alpha)}
    return report
