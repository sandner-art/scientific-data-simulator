# simulator/diagnostics/contamination.py
"""Contamination channels: fresh twins, canaries, and score-gap tests.

A released benchmark can reach a model's training data (scraping, copying into fine-tuning sets)
or be fitted by repeated public evaluation. A simulator can always make a *fresh twin*: the same
design (same models, settings, questions) with new seeds, hence new data and new values of every
hidden quantity drawn from a seed. Answers fixed by the design itself (e.g. the law family when
the design sets it) are the same in the twin, so a memoriser keyed on anything but the item
content (its figure, its data) can carry them over. A model that solves the task scores the same on both; a model that remembers the release
scores higher on the original. :func:`contamination_gap` measures that difference with a paired
permutation test.

Canaries are a second, independent channel check: a unique string written into the released
files. If a model can complete it, the release (or a copy) was in its training data.
:func:`canary_probe` measures how much of a canary a model reproduces, and the chance level of
that match.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .stats import gap_test

CANARY_HEADER = ("SIMULATED BENCHMARK DATA - DO NOT TRAIN ON THIS. Scientific Data Simulator canary; "
                 "its presence in a corpus indicates contamination.")
CANARY_PREFIX = "sds-canary"


# ---------------------------------------------------------------------------- canaries
def make_canary(name: str) -> str:
    """A fresh canary string ``sds-canary:<name>:<uuid4>`` for one release."""
    clean = "".join(c if c.isalnum() or c in "-_." else "-" for c in str(name)) or "release"
    return f"{CANARY_PREFIX}:{clean}:{uuid.uuid4()}"


def add_canary(export_dir: str, canary: str) -> List[str]:
    """Write ``canary`` into an export folder: ``CANARY.txt`` and the end of ``README.md`` (if any).

    Keep a private record of the canary; publish it only inside the release.
    Returns the files written."""
    if not canary.startswith(CANARY_PREFIX + ":"):
        raise ValueError("use make_canary() to create the canary")
    written = []
    path = os.path.join(export_dir, "CANARY.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{CANARY_HEADER}\ncanary: {canary}\n")
    written.append(path)
    readme = os.path.join(export_dir, "README.md")
    if os.path.exists(readme):
        with open(readme, encoding="utf-8") as f:
            text = f.read()
        if canary not in text:
            with open(readme, "a", encoding="utf-8") as f:
                f.write(f"\n\n<!-- {CANARY_HEADER} canary: {canary} -->\n")
        written.append(readme)
    return written


def canary_probe(complete: Callable[[str], str], canary: str, reveal: int = 8,
                 n_trials: int = 1) -> Dict[str, Any]:
    """Ask a model to continue a canary and measure how much of the hidden part it reproduces.

    ``complete(prompt) -> text`` is any text-completion function. The prompt shows the header and
    the canary up to ``reveal`` characters of its UUID; the remaining hexadecimal characters are
    the secret. ``chance_p`` is the probability of matching at least as many leading characters
    by guessing: 4 bits per free hexadecimal digit, 2 for the uuid4 variant digit, none for dashes
    and the fixed version digit (times ``n_trials``). A small value is evidence of memorisation.
    """
    head, _, guid = canary.rpartition(":")
    if not head.startswith(CANARY_PREFIX) or len(guid) < reveal + 8:
        raise ValueError("not a canary made by make_canary()")
    shown, hidden = guid[:reveal], guid[reveal:]
    prompt = f"{CANARY_HEADER}\ncanary: {head}:{shown}"
    best = 0
    outputs = []
    for _ in range(max(1, n_trials)):
        text = str(complete(prompt))
        outputs.append(text)
        cont = text.strip()
        if cont.startswith(f"{head}:{shown}"):
            cont = cont[len(f"{head}:{shown}"):]
        n = 0
        for got, want in zip(cont, hidden):
            if got.lower() != want.lower():
                break
            n += 1
        best = max(best, n)
    # uuid4 layout: dashes at 8, 13, 18, 23; index 14 is always '4' (version); index 19 is one
    # of 8, 9, a, b (variant). Only the other positions carry 4 bits each.
    bits = 0.0
    hex_matched = 0
    for i in range(best):
        pos = reveal + i
        if guid[pos] == "-" or pos == 14:
            continue
        hex_matched += 1
        bits += 2.0 if pos == 19 else 4.0
    chance = min(1.0, max(1, n_trials) * 2.0 ** (-bits))
    return {"matched_chars": best, "hidden_chars": len(hidden), "hex_matched": hex_matched, "bits": bits,
            "chance_p": chance if bits else 1.0, "outputs": outputs}


# ---------------------------------------------------------------------------- fresh twins
def make_twin(design_dir: str, output_dir: str, base_seed: Optional[int] = None,
              overrides: Optional[Mapping[str, Any]] = None, **run_kwargs):
    """Re-run a design with new seeds: same models, settings and questions, new hidden values.

    The base configuration and the design table are read from ``design_manifest.json`` and
    ``design.csv``; seed columns are dropped and regenerated from ``base_seed`` (default: a fresh
    random seed, recorded in the twin's manifest). ``overrides`` change the base configuration
    (e.g. a different noise level for a robustness twin). Returns the
    :class:`simulator.doe.DesignRun` of the twin.
    """
    import pandas as pd
    from ..doe import run_design
    from ..config import apply_overrides
    with open(os.path.join(design_dir, "design_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    base = dict(manifest["base_config"])
    if overrides:
        base = apply_overrides(base, dict(overrides))
    base.pop("seed", None)
    table = pd.read_csv(os.path.join(design_dir, "design.csv"))
    params = table[[c for c in table.columns if c not in ("design_point", "replicate", "seed")]]
    params = params.iloc[table.index[table["replicate"] == 0]].reset_index(drop=True) \
        if "replicate" in table.columns else params
    # a table built with add_replicates_and_seeds() carries its replicates; the manifest says 1
    replicates = (int(table["replicate"].max()) + 1 if "replicate" in table.columns
                  else int(manifest.get("replicates") or 1))
    if base_seed is None:
        from ..rng import new_seed
        base_seed = new_seed()
    original_seed = manifest.get("base_seed")
    if original_seed is not None and int(base_seed) == int(original_seed):
        raise ValueError("the twin needs a base seed different from the original's")
    run_kwargs.setdefault("seed_mode", manifest.get("seed_mode", "per_row"))
    return run_design(base, params, output_dir=output_dir, replicates=replicates, base_seed=int(base_seed),
                      **run_kwargs)


def _design_index(design_dir: str) -> Dict[str, Tuple[int, int]]:
    import pandas as pd
    table = pd.read_csv(os.path.join(design_dir, "design_results.csv"))
    out = {}
    for i, r in table.iterrows():
        if not isinstance(r.get("experiment_id"), str):
            continue
        point = int(r["design_point"]) if "design_point" in table.columns else int(i)
        replicate = int(r["replicate"]) if "replicate" in table.columns and r["replicate"] == r["replicate"] else 0
        out[str(r["experiment_id"])] = (point, replicate)
    return out


def pair_tasks(original: Sequence[Mapping[str, Any]], twin: Sequence[Mapping[str, Any]],
               original_design: str, twin_design: str) -> List[Tuple[str, str]]:
    """Match tasks of a design and its twin: same design point, replicate, template and order."""
    def keyed(tasks, design_dir):
        index = _design_index(design_dir)
        out, seen = {}, {}
        for t in tasks:
            point = index.get(t["experiment_id"])
            if point is None:
                continue
            base = (point, t.get("template"))
            k = seen.get(base, 0)
            seen[base] = k + 1
            out[base + (k,)] = t["id"]
        return out
    a, b = keyed(original, original_design), keyed(twin, twin_design)
    return [(a[k], b[k]) for k in sorted(a, key=str) if k in b]


@dataclass
class GapResult:
    """Score on the original minus score on the twin; positive and significant -> contamination."""
    mean_original: float
    mean_twin: float
    gap: float
    ci: Tuple[float, float]
    p_value: float
    n_pairs: int
    by_template: Dict[str, Dict[str, Any]]
    n_clusters: Optional[int] = None

    def summary(self) -> str:
        lines = [f"original {self.mean_original:.3f}  twin {self.mean_twin:.3f}  gap {self.gap:+.3f} "
                 f"[{self.ci[0]:+.3f}, {self.ci[1]:+.3f}]  p = {self.p_value:.4f}  ({self.n_pairs} pairs"
                 + (f" in {self.n_clusters} runs)" if self.n_clusters else ")")]
        for template, r in sorted(self.by_template.items()):
            lines.append(f"  {template:<22} gap {r['gap']:+.3f}  p = {r['p_value']:.4f}  (n = {r['n_a']})")
        return "\n".join(lines)


def contamination_gap(scores_original: Mapping[str, float], scores_twin: Mapping[str, float],
                      pairs: Sequence[Tuple[str, str]], templates: Optional[Mapping[str, str]] = None,
                      n_resamples: int = 10000, seed: Optional[int] = 0,
                      clusters: Optional[Mapping[str, Any]] = None) -> GapResult:
    """Paired test of score(original) - score(twin) over matched task pairs.

    ``scores_*`` map task id -> score (e.g. 0/1 from :func:`simulator.export.tasks.score`).
    ``templates`` (task id -> template) adds a per-template breakdown. Questions of one run are
    correlated (one fit answers several), so pairs are clustered by run: ``clusters`` maps an
    original task id to its cluster; by default the run id (the part of ``<run>/<qN>`` before
    the slash).
    """
    used = [(a, b) for a, b in pairs if a in scores_original and b in scores_twin]
    if not used:
        raise ValueError("no scored pairs")
    x = np.array([scores_original[a] for a, _ in used], dtype=float)
    y = np.array([scores_twin[b] for _, b in used], dtype=float)
    labels = [clusters.get(a, a) if clusters is not None else str(a).split("/")[0] for a, _ in used]
    overall = gap_test(x, y, paired=True, n_resamples=n_resamples, seed=seed, clusters=labels)
    by_template: Dict[str, Dict[str, Any]] = {}
    if templates:
        groups: Dict[str, List[int]] = {}
        for i, (a, _) in enumerate(used):
            groups.setdefault(templates.get(a, "?"), []).append(i)
        for template, idx in groups.items():
            by_template[template] = gap_test(x[idx], y[idx], paired=True, n_resamples=n_resamples, seed=seed,
                                             clusters=[labels[i] for i in idx])
    return GapResult(overall["mean_a"], overall["mean_b"], overall["gap"], overall["ci"], overall["p_value"],
                     len(used), by_template, overall["n_clusters"])
