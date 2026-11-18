# scidatasim/diagnostics/shortcuts.py
"""Shortcut artefacts: can the answers be predicted without the data?

A generated benchmark has a *shortcut* when some cue that is not the evidence predicts the
answer: the answer distribution itself (a majority class, a typical value), the position of the
correct choice, or public metadata such as the noise level, the number of points, the length of
the caption or the size of the figure file. In simulated causal benchmarks the classic case is
*varsortability* (Reisach et al., 2021): marginal variance increases along the causal order, so
sorting by variance "discovers" the graph.

* :func:`blind_baseline` answers every question from the *other* questions' answers alone
  (leave-one-out): majority choice, majority position, median number, best constant number, most
  common formula.
* :func:`feature_shortcuts` answers from public per-run features with a leave-one-out
  nearest-neighbour predictor, and tests with a permutation test whether the features carry
  information about the answers beyond the blind baseline.
* :func:`public_features` collects features that are public in a bundle release: figure file
  size, caption length, number of public table rows.
* :func:`approximator_baseline` tests the *scorer*: it answers formula questions with a
  polynomial fitted to the run's observations. A scorer that accepts it rewards curve fitting,
  not discovery of the law (a "scorer loophole"; the same holds for any score a flexible model
  can meet without the mechanism).

A solver that beats chance only by such cues is not reasoning about the data; a generator whose
answers are predictable from them needs fixing (or the cue withheld).
"""
from __future__ import annotations

import numbers
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


def _constant_baseline(group: Sequence[Mapping[str, Any]], n_candidates: int = 101) -> Tuple[float, Optional[float]]:
    """Leave-one-out accuracy of the best constant answer, and the constant best on all items.

    Candidates are 0 and quantiles of the answers. For each item, the constant that is correct
    most often on the *other* items is tried on it. This finds templates whose tolerances let one
    number pass many questions: extrapolations of laws that have decayed to almost 0 pass with 0
    when the tolerance has a floor tied to the observed scale.
    """
    values = np.array([float(t["answer"]) for t in group if isinstance(t["answer"], numbers.Real)
                       and not isinstance(t["answer"], bool) and np.isfinite(float(t["answer"]))])
    if values.size < 2:
        return 0.0, None
    candidates = np.unique(np.concatenate([[0.0], np.quantile(values, np.linspace(0, 1, n_candidates))]))
    if group[0]["answer_type"] == "integer":
        candidates = np.unique(np.round(candidates))
    hits = np.array([[_check(t, int(c) if t["answer_type"] == "integer" else float(c)) for t in group]
                     for c in candidates], dtype=float)  # candidates x items
    totals = hits.sum(axis=1)
    loo = 0
    for i in range(len(group)):
        best = int(np.argmax(totals - hits[:, i]))  # chosen without item i
        loo += hits[best, i]
    return float(loo / len(group)), float(candidates[int(np.argmax(totals))])


def blind_baseline(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Leave-one-out accuracy of answering each template from the other answers only.

    ``tasks`` must carry answers (private tasks, e.g. from
    :func:`scidatasim.export.tasks.collect_bundle_tasks`). Per template: ``accuracy`` (majority
    choice / median number / most common text), for numbers also ``constant_accuracy`` (the best
    constant answer, chosen on the other items; ``constant``: the one best on all items),
    ``position_accuracy`` (choice questions: the most
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
        if entry["answer_type"] in ("number", "integer"):
            entry["constant_accuracy"], entry["constant"] = _constant_baseline(group)
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


def polynomial_formula(x, y, max_degree: int = 8) -> Optional[str]:
    """A polynomial in x fitted to (x, y) by least squares, its degree (1..``max_degree``) chosen
    by BIC, written as a formula. ``None`` if there are too few finite points."""
    import warnings
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    best = None
    for degree in range(1, min(int(max_degree), x.size - 2) + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # badly conditioned high degrees: BIC rejects them anyway
            coefficients = np.polyfit(x, y, degree)
        rss = float(np.sum((np.polyval(coefficients, x) - y) ** 2)) + 1e-300
        bic = x.size * np.log(rss / x.size) + (degree + 1) * np.log(x.size)
        if best is None or bic < best[0]:
            best = (bic, coefficients)
    if best is None:
        return None
    degree = len(best[1]) - 1
    return " + ".join(f"({c:.17g})*x**{degree - i}" for i, c in enumerate(best[1]))


def approximator_baseline(tasks: Sequence[Mapping[str, Any]], observations: Mapping[str, Any],
                          max_degree: int = 8) -> Dict[str, Dict[str, Any]]:
    """Accuracy of a flexible approximator on the expression (formula) questions.

    For each task with ``answer_type: expression`` the run's observations
    (``observations[task["experiment_id"]] = (x, y)``) are fitted with
    :func:`polynomial_formula` and the polynomial is scored like any answer
    (:func:`scidatasim.bundle.check_answer`). ``tasks`` must carry answers.

    Returns, per template, ``n``, ``accuracy`` and the ids of the tasks a polynomial ``passed``.
    Anything well above zero means the formula score can be met without the law. Polynomials
    are the approximator because the formula grammar has no splines or piecewise terms.
    """
    from ..bundle import check_answer
    out: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        if task.get("answer_type") != "expression" or task.get("experiment_id") not in observations:
            continue
        x, y = observations[task["experiment_id"]]
        formula = polynomial_formula(x, y, max_degree)
        entry = out.setdefault(task.get("template", "?"), {"n": 0, "passed": []})
        entry["n"] += 1
        if formula is not None and check_answer(task, formula):
            entry["passed"].append(task["id"])
    for entry in out.values():
        entry["accuracy"] = len(entry["passed"]) / entry["n"]
    return out
