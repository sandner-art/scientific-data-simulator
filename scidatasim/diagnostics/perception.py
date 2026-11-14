# scidatasim/diagnostics/perception.py
"""Perception against reasoning: the same items drawn in different figure styles.

:func:`scidatasim.doe.cross_styles` runs every design point in several figure styles with the same
seed, so data, questions and answers are identical and only the picture changes. A drop in
accuracy from one style to another is then a perception effect (reading the chart), not a
reasoning one. :func:`style_gap` measures it on matched items with a paired test.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from .stats import gap_test


def _item_key(task: Mapping[str, Any]):
    # the same seed and the same question id: one item drawn in several styles
    return task.get("seed"), str(task["id"]).rsplit("/", 1)[-1]


def style_gap(tasks: Sequence[Mapping[str, Any]], correct: Mapping[str, bool], base: str,
              key: str = "figure_variant", n_resamples: int = 10000, seed: Optional[int] = 0
              ) -> Dict[str, Dict[str, Any]]:
    """Accuracy of each figure variant minus that of ``base``, on the items both contain.

    ``tasks`` are task records with ``seed`` and ``figure_variant`` (from
    :func:`scidatasim.export.tasks.collect_bundle_tasks`); ``correct`` maps task ids to right or
    wrong. Pairs share a seed and a question id; the paired permutation test clusters by run
    (questions of one run are not independent). Returns ``{variant: {"n_pairs", "accuracy",
    "base_accuracy", "difference", "p_value", "interval"}}``.
    """
    by_variant: Dict[str, Dict[Any, bool]] = {}
    for t in tasks:
        variant = t.get(key)
        if variant is None or t["id"] not in correct:
            continue
        by_variant.setdefault(str(variant), {})[_item_key(t)] = bool(correct[t["id"]])
    if base not in by_variant:
        raise ValueError(f"no scored tasks of the base variant {base!r}")
    out: Dict[str, Dict[str, Any]] = {}
    for variant, hits in sorted(by_variant.items()):
        if variant == base:
            continue
        shared = sorted(set(hits) & set(by_variant[base]), key=str)
        if not shared:
            continue
        a = [float(hits[k]) for k in shared]
        b = [float(by_variant[base][k]) for k in shared]
        test = gap_test(a, b, paired=True, n_resamples=n_resamples, seed=seed, clusters=[k[0] for k in shared])
        out[variant] = {"n_pairs": len(shared), "accuracy": sum(a) / len(a), "base_accuracy": sum(b) / len(b),
                        "difference": test["gap"], "p_value": test["p_value"], "interval": test["ci"]}
    return out
