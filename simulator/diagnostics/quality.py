# simulator/diagnostics/quality.py
"""Quality card for a generated question set: balance, diversity, duplicates, solvability.

Synthetic benchmarks fail quietly: a choice question whose answer is almost always the same
option, numeric answers that barely vary, the same question asked twice with the same answer, or
questions that even a solver with the noise-free truth cannot answer within tolerance (not
identifiable from the data). :func:`quality_card` measures these per template:

* ``majority_share`` / ``answer_entropy`` (choices: normalised entropy, 1 = balanced);
* ``position_p`` (choices: chi-square p-value that the correct position is uniform);
* ``distinct_ratio`` and ``relative_spread`` (numbers: interquartile range / median magnitude);
* ``duplicates`` (identical prompt and answer within the template);
* ``fidelity`` counts (reference / audit / ...);
* ``solver_accuracy`` (optional): accuracy of a reference solver ``solver(task) -> response``.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np


def _chi2_uniform_p(counts: Sequence[int], n_sim: int = 20000, seed: int = 0) -> float:
    """p-value for "uniform over positions": chi-square, or Monte Carlo when an expected count
    is below 5 (the chi-square approximation is poor there)."""
    counts = np.asarray(counts, dtype=float)
    k, n = counts.size, counts.sum()
    if k < 2 or n == 0:
        return 1.0
    expected = n / k
    stat = float(np.sum((counts - expected) ** 2 / expected))
    if expected >= 5:
        from scipy.stats import chi2
        return float(chi2.sf(stat, k - 1))
    sims = np.random.default_rng(seed).multinomial(int(n), [1.0 / k] * k, size=n_sim)
    sim_stats = np.sum((sims - expected) ** 2 / expected, axis=1)
    return float((1 + np.sum(sim_stats >= stat - 1e-12)) / (n_sim + 1))


def quality_card(tasks: Sequence[Mapping[str, Any]], solver: Optional[Callable[[Mapping[str, Any]], Any]] = None
                 ) -> Dict[str, Any]:
    """Quality measures per template and overall (``tasks`` must carry answers)."""
    from ..bundle import check_answer
    groups: Dict[str, list] = {}
    for t in tasks:
        groups.setdefault(str(t.get("template")), []).append(t)
    card: Dict[str, Any] = {"n_tasks": len(tasks), "n_runs": len({t["experiment_id"] for t in tasks}),
                            "fidelity": dict(Counter(str(t.get("fidelity")) for t in tasks)), "templates": {}}
    for template, group in sorted(groups.items()):
        kind = group[0]["answer_type"]
        entry: Dict[str, Any] = {"n": len(group), "answer_type": kind,
                                 "answerable_from": dict(Counter(str(t.get("answerable_from")) for t in group))}
        answers = [t["answer"] for t in group]
        keys = Counter((t.get("prompt") or t.get("question"), str(t["answer"])) for t in group)
        entry["duplicates"] = int(sum(c - 1 for c in keys.values() if c > 1))
        if kind == "choice":
            counts = Counter(str(a) for a in answers)
            probs = np.array(list(counts.values()), dtype=float) / len(answers)
            n_options = max(len(t.get("choices") or []) for t in group)
            entry["majority_share"] = float(probs.max())
            entry["answer_entropy"] = float(-(probs * np.log(probs)).sum() / math.log(max(2, len(
                {str(c) for t in group for c in (t.get("choices") or [])}))))
            positions = Counter(list(t["choices"]).index(t["answer"]) for t in group
                                if t.get("choices") and t["answer"] in t["choices"])
            entry["position_counts"] = [positions.get(i, 0) for i in range(n_options)]
            uniform = len({len(t.get("choices") or []) for t in group}) == 1
            entry["position_p"] = _chi2_uniform_p(entry["position_counts"]) if uniform else None
        elif kind in ("number", "integer"):
            values = np.array([float(a) if isinstance(a, (int, float, np.number)) and not isinstance(a, bool)
                               else np.nan for a in answers], dtype=float)
            entry["non_finite"] = int(np.sum(~np.isfinite(values)))
            values = values[np.isfinite(values)]
            if values.size == 0:
                card["templates"][template] = entry
                continue
            entry["distinct_ratio"] = float(len(np.unique(np.round(values, 12))) / len(values))
            q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            entry["relative_spread"] = float((q3 - q1) / max(abs(med), 1e-12))
            entry["range"] = [float(values.min()), float(values.max())]
        else:
            entry["distinct_ratio"] = len({str(a) for a in answers}) / len(answers)
        if solver is not None:
            hits = 0
            for t in group:
                try:
                    hits += bool(check_answer(t, solver(t)))
                except Exception:
                    pass
            entry["solver_accuracy"] = hits / len(group)
        card["templates"][template] = entry
    return card


def card_markdown(card: Mapping[str, Any]) -> str:
    """The quality card as a Markdown table."""
    lines = [f"Tasks: {card['n_tasks']} from {card['n_runs']} runs; fidelity: "
             + ", ".join(f"{k} {v}" for k, v in sorted(card["fidelity"].items())), "",
             "| template | n | type | balance / spread | position p | duplicates | solver |",
             "|---|---|---|---|---|---|---|"]
    for template, e in card["templates"].items():
        if e["answer_type"] == "choice":
            balance = f"majority {e['majority_share']:.2f}, entropy {e['answer_entropy']:.2f}"
        elif "relative_spread" in e:
            balance = f"IQR/median {e['relative_spread']:.2f}, distinct {e['distinct_ratio']:.2f}"
        elif "distinct_ratio" in e:
            balance = f"distinct {e['distinct_ratio']:.2f}"
        else:
            balance = "no finite answers"
        pos = e.get("position_p")
        solver = e.get("solver_accuracy")
        lines.append(f"| {template} | {e['n']} | {e['answer_type']} | {balance} | "
                     f"{'' if pos is None else f'{pos:.3f}'} | {e['duplicates']} | "
                     f"{'' if solver is None else f'{solver:.2f}'} |")
    return "\n".join(lines)
