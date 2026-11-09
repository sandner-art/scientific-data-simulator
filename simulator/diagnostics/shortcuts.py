# simulator/diagnostics/shortcuts.py
"""Shortcut artefacts: can the answers be predicted without the data?

A generated benchmark has a *shortcut* when some cue that is not the evidence predicts the
answer: the answer distribution itself (a majority class, a typical value), the position of the
correct choice, or public metadata such as the noise level, the number of points, the length of
the caption or the size of the figure file. In simulated causal benchmarks the classic case is
*varsortability* (Reisach et al., 2021): marginal variance increases along the causal order, so
sorting by variance "discovers" the graph.

* :func:`blind_baseline` answers every question from the *other* questions' answers alone
  (leave-one-out): majority choice, majority position, median number, most common formula.
* :func:`feature_shortcuts` answers from public per-run features with a leave-one-out
  nearest-neighbour predictor, and tests with a permutation test whether the features carry
  information about the answers beyond the blind baseline.
* :func:`public_features` collects features that are public in a bundle release: figure file
  size, caption length, number of public table rows.

A solver that beats chance only by such cues is not reasoning about the data; a generator whose
answers are predictable from them needs fixing (or the cue withheld).
"""
from __future__ import annotations

import numbers
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .stats import binomial_p_greater


def _check(task: Mapping[str, Any], response: Any) -> bool:
    from ..bundle import check_answer
    try:
        return bool(check_answer(task, response))
    except Exception:
        return False


def _by_template(tasks: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for t in tasks:
        groups.setdefault(str(t.get("template")), []).append(t)
    return groups


def _blind_guess(task: Mapping[str, Any], others: Sequence[Mapping[str, Any]], mode: str) -> Any:
    answers = [o["answer"] for o in others]
    if not answers:
        return None
    kind = task["answer_type"]
    if kind == "choice":
        choices = list(task.get("choices") or [])
        if mode == "position":
            positions = Counter(list(o["choices"]).index(o["answer"]) for o in others
                                if o.get("choices") and o["answer"] in o["choices"])
            for pos, _ in positions.most_common():
                if pos < len(choices):
                    return choices[pos]
            return choices[0] if choices else None
        counts = Counter(str(a) for a in answers)
        ranked = [c for c in choices if str(c) in counts]
        return max(ranked, key=lambda c: counts[str(c)]) if ranked else (choices[0] if choices else None)
    if kind in ("number", "integer"):
        values = np.array([float(a) for a in answers if isinstance(a, numbers.Real) and not isinstance(a, bool)
                           and np.isfinite(float(a))])
        if values.size == 0:
            return None
        guess = float(np.median(values))
        return int(round(guess)) if kind == "integer" else guess
    return Counter(str(a) for a in answers).most_common(1)[0][0]


def _chance(task: Mapping[str, Any]) -> Optional[float]:
    if task["answer_type"] == "choice" and task.get("choices"):
        return 1.0 / len(task["choices"])
    return None


def blind_baseline(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Leave-one-out accuracy of answering each template from the other answers only.

    ``tasks`` must carry answers (private tasks, e.g. from
    :func:`simulator.export.tasks.collect_bundle_tasks`). Per template: ``accuracy`` (majority
    choice / median number / most common text), ``position_accuracy`` (choice questions: the most
    common correct position), ``chance`` (choice questions: mean 1/len(choices)), and ``p_value``
    of the better blind accuracy against chance (exact binomial, Bonferroni-corrected for the two
    strategies). Leave-one-out majority voting is biased *downwards* on perfectly balanced
    labels (removing an item makes its class the minority), so a low value there is not evidence
    of balance by itself; ``quality_card`` reports the balance directly.
    """
    report: Dict[str, Dict[str, Any]] = {}
    for template, group in _by_template(tasks).items():
        hits, pos_hits = 0, 0
        for i, task in enumerate(group):
            others = group[:i] + group[i + 1:]
            hits += _check(task, _blind_guess(task, others, "value"))
            if task["answer_type"] == "choice":
                pos_hits += _check(task, _blind_guess(task, others, "position"))
        n = len(group)
        entry: Dict[str, Any] = {"n": n, "answer_type": group[0]["answer_type"], "accuracy": hits / n}
        chances = [c for c in (_chance(t) for t in group) if c is not None]
        if chances:
            chance = float(np.mean(chances))
            # two blind strategies are tried: Bonferroni-corrected
            entry.update(position_accuracy=pos_hits / n, chance=chance,
                         p_value=min(1.0, 2 * binomial_p_greater(max(hits, pos_hits), n, chance)))
        report[template] = entry
    return report


def _knn_guess(task, neighbours: Sequence[Mapping[str, Any]]) -> Any:
    return _blind_guess(task, neighbours, "value")


def feature_shortcuts(tasks: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]],
                      k: int = 5, n_permutations: int = 200, seed: Optional[int] = 0,
                      templates: Optional[Sequence[str]] = None,
                      groups: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Leave-one-out k-nearest-neighbour answers from public per-run features.

    ``features`` maps experiment id -> {feature: value} (numbers; standardised internally). For
    each task, the ``k`` runs nearest in feature space (same template, other runs) vote (choices,
    text) or give their median (numbers). ``p_value`` comes from a permutation test that shuffles
    the features across runs: a small value means the features predict the answers.

    ``groups`` maps experiment id -> group (e.g. the design point). Neighbours from the same group
    are left out, so a feature that merely identifies replicates of one design point - which share
    design-level answers - is not mistaken for a shortcut. Default: each run is its own group.
    """
    rng = np.random.default_rng(seed)
    names = sorted({f for row in features.values() for f in row})
    if not names:
        raise ValueError("no features")
    ids = sorted(features)
    matrix = np.array([[float(features[i].get(f, np.nan)) for f in names] for i in ids], dtype=float)
    col_mean = np.nanmean(matrix, axis=0)
    col_std = np.nanstd(matrix, axis=0)
    col_std[~np.isfinite(col_std) | (col_std == 0)] = 1.0
    matrix = np.where(np.isfinite(matrix), (matrix - col_mean) / col_std, 0.0)
    row_of = {eid: i for i, eid in enumerate(ids)}
    group_of = {eid: (groups.get(eid, eid) if groups is not None else eid) for eid in ids}

    def accuracy(group, mat) -> float:
        rows = [row_of[t["experiment_id"]] for t in group]
        hits = 0
        for i, task in enumerate(group):
            d = np.linalg.norm(mat[rows] - mat[rows[i]], axis=1)
            d[i] = np.inf
            mine = group_of[task["experiment_id"]]
            same = [j for j, t in enumerate(group) if group_of[t["experiment_id"]] == mine]
            d[same] = np.inf
            order = [j for j in np.argsort(d, kind="stable") if np.isfinite(d[j])][:k]
            hits += _check(task, _knn_guess(task, [group[j] for j in order]))
        return hits / len(group)

    report: Dict[str, Dict[str, Any]] = {}
    for template, group in _by_template(tasks).items():
        if templates is not None and template not in templates:
            continue
        group = [t for t in group if t["experiment_id"] in row_of]
        if len(group) <= k:
            continue
        observed = accuracy(group, matrix)
        null = []
        for _ in range(n_permutations):
            null.append(accuracy(group, matrix[rng.permutation(len(ids))]))
        null = np.asarray(null)
        blind = blind_baseline(group)[template]["accuracy"]
        report[template] = {"n": len(group), "accuracy": observed, "blind_accuracy": blind,
                            "null_mean": float(null.mean()),
                            "gain": observed - float(null.mean()),
                            "p_value": float((1 + np.sum(null >= observed - 1e-12)) / (n_permutations + 1)),
                            "features": names}
    return report


def public_features(run_dirs: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Per-run cues visible in a bundle release (without reading the data): ``figure_bytes``,
    ``caption_chars``, ``table_rows`` (if the bundle has a public table)."""
    import json
    from ..export.tasks import _run_dirs
    out: Dict[str, Dict[str, float]] = {}
    for run_dir in _run_dirs(list(run_dirs) if not isinstance(run_dirs, str) else run_dirs):
        with open(os.path.join(run_dir, "experiment_record.json"), encoding="utf-8") as f:
            eid = json.load(f)["experiment_id"]
        bundle = os.path.join(run_dir, "bundle")
        row: Dict[str, float] = {}
        figure = next((os.path.join(bundle, n) for n in sorted(os.listdir(bundle)) if n.startswith("figure.")),
                      None) if os.path.isdir(bundle) else None
        if figure:
            row["figure_bytes"] = float(os.path.getsize(figure))
        caption = os.path.join(bundle, "caption.txt")
        if os.path.exists(caption):
            with open(caption, encoding="utf-8") as f:
                row["caption_chars"] = float(len(f.read()))
        meta = os.path.join(bundle, "bundle.json")
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as f:
                files = json.load(f).get("files", {})
            table = files.get("table.csv", {})
            if table.get("visibility") == "public" and os.path.exists(os.path.join(bundle, "table.csv")):
                with open(os.path.join(bundle, "table.csv"), encoding="utf-8") as f:
                    row["table_rows"] = float(sum(1 for _ in f) - 1)
        out[eid] = row
    return out
