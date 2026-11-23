"""Audit SciLaws-Bench with the simulator's instruments: a leak audit of the release, and generic
non-law fits scored by the benchmark's own numeric scorer.

SciLaws-Bench (Huang et al., arXiv:2609.01552) scores a submitted law by S_N, its test error
relative to the best published law (0.5: as good as the best published law, 1: perfect), and
separately by S_V, rubrics judged by a code agent behind an anti-hacking gate. Two measurements on
the public release:

1. **Leak audit.** ``scidatasim.audit.audit_export`` on each task, with an answer key built from the
   task's hidden law (``simulator/formula.py``, parsed with ``ast``; never imported or run) and the
   published laws' fitted constants (``eval/reference_metrics.json``): on the solver-facing files
   (``metadata.yaml``, ``data/``), and - the hidden law only - on the grader-facing ones (``eval/``,
   ``simulator/``) as a positive control, since the release documents that those are answer keys.
   Also: whether ``simulator/state.joblib`` carries the law's text and which modules unpickling it
   would import (its pickle opcodes are listed after decompression; it is never unpickled), rows
   shared by the training and test files, how far the test inputs lie outside the training range,
   and whether any file of the release carries a canary string.
2. **Non-law submissions.** Families any curve fitter tries - a constant, polynomials and additive
   natural cubic splines, with raw or log target and inputs; for Type II also per-group
   polynomials and one global shape with a per-group offset and/or scale - fitted on
   ``train.csv`` only, the variant chosen by cross-validation on it (folds at the edge of the
   training range when the test inputs lie outside it; for Type II, held-out groups split as the
   test groups are), written in the benchmark's submission format and scored by its harness
   (``harness/evaluate_numeric.py score``, in a child process). Each submission is held to three
   limits on its fitted numbers: what the scorer checks (the lengths of ``LAW_CONSTANTS`` and
   ``LOCAL_FITTABLE``: numbers written into ``predict`` are not counted), the anti-hacking judge's
   rule (fitted numbers up to the caps + 3, ``harness/evaluate_validity.py``) and the task
   prompt's rule (up to the caps).

Reports, per limit, the share of tasks where a non-law reaches S_N >= 0.5 and its mean S_N next to
the agents' per-task scores published with the harness (``baseline_agent/model_task_scores.json``),
by task type and by split (interpolation, extrapolation, new groups). S_V is not computed: it needs
the code-agent judge.

Nothing from the release is executed: the only code that runs is the harness scorer, on
submissions this script writes. Windows has no ``signal.alarm``, which the scorer uses for its
per-group fit timeout; there the timeout is disabled, and the fit times of the submissions are
measured here and compared with the cap instead.

    hf download RealSR/SciLaws-Bench --repo-type dataset --local-dir scilaws_data --include "tasks/*"
    git clone https://github.com/yiyihum/SciLaws-Bench scilaws_code
    git -C scilaws_code checkout 8174c7b5289fd3e414892f20efca322cb1e0e5e9
    python tools/audit_scilaws.py --tasks scilaws_data/tasks --harness scilaws_code --out experiments_output/scilaws

``--no-score`` skips the harness and reports S_N computed here (the harness's formula; checked
against the harness whenever both run). ``--only`` takes task names, ``--skip-audit`` and
``--skip-fits`` leave out a part, ``--resume`` keeps finished tasks. Writes
``scilaws_audit.{json,md}``, one JSON per task (``tasks/``) and the submissions
(``submissions/<task>/<name>.py``) to ``--out``.
"""
from __future__ import annotations

import argparse
import ast
import bz2
import datetime as _dt
import hashlib
import importlib.util
import inspect
import itertools
import json
import lzma
import math
import os
import pickletools
import platform
import re
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PINNED_HARNESS = "8174c7b5289fd3e414892f20efca322cb1e0e5e9"
DATASET = "RealSR/SciLaws-Bench (Hugging Face dataset), tasks/"
PASS = 0.5  # S_N of the best published law
NEAR = 0.45  # S_N of an error 10 % above the best published law's
JUDGE_SLACK = 3  # harness/evaluate_validity.py, CONSTANT_DISCIPLINE_RUBRIC: "up to +3 slack"
TIERS = ("scorer", "judge", "prompt")
TIER_TEXT = {"scorer": "what the scorer checks", "judge": "the anti-hacking judge's rule (caps + 3)",
             "prompt": "the prompt's rule (caps)"}
MAX_CV_ROWS = 50_000
MAX_FIT_ROWS = 200_000
MAX_CV_GROUPS = 200
SEED = 20261120
HIGHER_BETTER = {"r2"}
PERFECT = {"r2": 1.0}
REF_EPS = 1e-4  # harness: a unit where the best reference is (near-)perfect is excluded

# The feature code below runs here (fitting, cross-validation, local scoring) and, copied by
# source, in every submission: the scored predictor is the fitted one.
def _z(X, log_in, center, scale):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    Z = np.empty_like(X)
    with np.errstate(all="ignore"):
        for j in range(X.shape[1]):
            u = np.log(X[:, j]) if log_in[j] else X[:, j]
            Z[:, j] = (u - center[j]) / scale[j]
    return np.where(np.isfinite(Z), Z, 0.0)


def _poly(Z, powers):
    F = np.ones((Z.shape[0], len(powers)))
    for k, p in enumerate(powers):
        for j, e in enumerate(p):
            if e:
                F[:, k] *= Z[:, j] ** e
    return F


def _ns(z, xi):
    K = len(xi)
    if K < 3:
        return z[:, None]

    def d(k):
        return (np.maximum(z - xi[k], 0.0) ** 3 - np.maximum(z - xi[K - 1], 0.0) ** 3) / (xi[K - 1] - xi[k])
    last = d(K - 2)
    return np.column_stack([z] + [d(k) - last for k in range(K - 2)])


def _basis(Z, kind, powers, knots, intercept):
    if kind == "poly":
        F = _poly(Z, powers)
    else:
        F = np.column_stack([np.ones(Z.shape[0])] + [_ns(Z[:, j], knots[j]) for j in range(Z.shape[1])])
    return F if intercept else F[:, 1:]


def _design(Z, mode, kind, powers, knots, coef):
    if mode in ("global", "group_poly"):
        return _basis(Z, kind, powers, knots, True)
    return _basis(Z, kind, powers, knots, mode == "scale") @ np.asarray(coef, dtype=float)


def _fit_local(mode, D, t, n_local):
    D = np.asarray(D, dtype=float)
    t = np.asarray(t, dtype=float)
    try:
        if mode == "group_poly":
            c = np.linalg.lstsq(D, t, rcond=None)[0]
        elif mode == "offset":
            c = np.array([np.mean(t - D)])
        elif mode == "scale":
            gg = float(D @ D)
            c = np.array([float(D @ t) / gg if gg > 0 else 1.0])
        else:
            c = np.linalg.lstsq(np.column_stack([np.ones_like(D), D]), t, rcond=None)[0]
        c = np.asarray(c, dtype=float)
        if c.shape != (n_local,) or not np.all(np.isfinite(c)):
            raise ValueError("degenerate fit")
    except Exception:
        c = np.zeros(n_local)
        mean_t = float(np.nan_to_num(np.mean(t))) if len(t) else 0.0
        mean_d = float(np.nan_to_num(np.mean(D))) if D.ndim == 1 and len(D) else 0.0
        if mode == "scale":
            c[0] = 1.0
        elif mode == "affine":
            c[:] = [mean_t - mean_d, 1.0]
        elif mode == "offset":
            c[0] = mean_t - mean_d
        else:
            c[0] = mean_t
    return np.nan_to_num(c)


def _apply_local(mode, D, c):
    if mode in ("global", "group_poly"):
        return D @ c
    if mode == "offset":
        return c[0] + D
    if mode == "scale":
        return c[0] * D
    return c[0] + c[1] * D


def _target(y, log):
    y = np.asarray(y, dtype=float)
    return np.log(np.clip(y, 1e-300, None)) if log else y


def _output(f, log):
    with np.errstate(all="ignore"):
        y = np.exp(np.clip(f, -700.0, 700.0)) if log else np.asarray(f, dtype=float)
    return np.nan_to_num(y, nan=0.0, posinf=1e300, neginf=-1e300)


FEATURE_FUNCTIONS = (_z, _poly, _ns, _basis, _design, _fit_local, _apply_local, _target, _output)


def feature_code() -> str:
    """The source of the feature functions, for the submissions."""
    return "\n\n".join(inspect.getsource(f).strip() for f in FEATURE_FUNCTIONS)


# The harness scorer, run in a child process. Windows has no SIGALRM: the per-group fit timeout
# becomes a no-op there (the fit times are measured by this script instead).
SHIM = r'''
import os, runpy, signal, sys
harness = sys.argv[1]
if not hasattr(signal, "SIGALRM"):
    _signal = signal.signal
    signal.SIGALRM = -1001
    signal.signal = lambda sig, handler: None if sig == -1001 else _signal(sig, handler)
    signal.alarm = lambda seconds: 0
script = os.path.join(harness, "harness", "evaluate_numeric.py")
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
'''


# ---------------------------------------------------------------------------- metrics (harness copy)
def metric(name: str, y_true, y_pred) -> Optional[float]:
    """The harness's metric registry (``harness/eval_formula.py``), on the finite pairs."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[keep], yp[keep]
    if yt.size == 0:
        return None
    with np.errstate(all="ignore"):
        e = yp - yt
        if name == "mse":
            return float(np.mean(e ** 2))
        if name == "rmse":
            return float(np.sqrt(np.mean(e ** 2)))
        if name == "mae":
            return float(np.mean(np.abs(e)))
        if name == "mdae":
            return float(np.median(np.abs(e)))
        if name == "r2":
            ss = float(np.sum((yt - yt.mean()) ** 2))
            return None if ss <= 0 else float(1.0 - float(np.sum(e ** 2)) / ss)
        if name == "smape":
            den = (np.abs(yt) + np.abs(yp)) / 2.0
            ok = den > 0
            return float(np.mean(np.abs(e[ok]) / den[ok])) if ok.any() else None
        if name == "mape":
            nz = yt != 0
            return float(np.mean(np.abs(e[nz] / yt[nz]))) if nz.any() else None
        if name == "log_mae":
            pos = yt > 0
            if not pos.any():
                return None
            return float(np.mean(np.abs(np.log10(np.clip(yp[pos], 1e-300, None)) - np.log10(yt[pos]))))
    raise ValueError(f"unknown metric {name!r}")


def unit_score(sub: float, ref: float, name: str) -> float:
    """The harness's per-unit score: 1 - 0.5 sub/ref (lower is better), clipped to [0, 1]."""
    if name in HIGHER_BETTER:
        s = 0.5 + 0.5 * (sub - ref) / (PERFECT[name] - ref)
    else:
        s = 1.0 - 0.5 * sub / ref
    return float(np.clip(s, 0.0, 1.0))


def _loss(name: str, value: Optional[float]) -> float:
    """Model-selection loss: log error (1 - r2 for r2); inf when undefined."""
    if value is None or not np.isfinite(value):
        return math.inf
    v = 1.0 - value if name in HIGHER_BETTER else value
    return math.log(max(v, 1e-300))


def best_reference(ref: Dict[str, Any], name: str) -> Optional[str]:
    """The harness's anchor: the reference baseline with the best mean per-unit metric."""
    cand = {}
    for key, b in (ref or {}).get("baselines", {}).items():
        if b.get("kind") != "reference":
            continue
        pc = b.get("per_cluster")
        if pc:
            vals = [m[name] for m in pc.values() if m and m.get(name) is not None]
            v = float(np.mean(vals)) if vals else None
        else:
            fm = b.get("metrics")
            v = float(fm[name]) if fm and fm.get(name) is not None else None
        if v is not None:
            cand[key] = v
    if not cand:
        return None
    return max(cand, key=cand.get) if name in HIGHER_BETTER else min(cand, key=cand.get)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


# ---------------------------------------------------------------------------- tasks
def find_tasks(root: str, only: Sequence[str] = ()) -> List[str]:
    """Task folders (with ``metadata.yaml`` and ``data/``) below ``root`` (or ``root`` itself)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if "metadata.yaml" in filenames and os.path.isdir(os.path.join(dirpath, "data")):
            found.append(dirpath)
            dirnames[:] = []
        elif os.path.relpath(dirpath, root).count(os.sep) >= 3:
            dirnames[:] = []
    if only:
        wanted = set(only)
        found = [p for p in found if os.path.basename(p) in wanted]
    return sorted(found)


def load_task(path: str) -> Dict[str, Any]:
    import pandas as pd
    import yaml
    with open(os.path.join(path, "metadata.yaml"), encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    data = os.path.join(path, "data")
    ttype = meta.get("type") or ("typeII" if os.path.exists(os.path.join(data, "test_fit.csv")) else "typeI")
    target = meta.get("target")
    target = target.get("name") if isinstance(target, dict) else target
    ref_path = os.path.join(path, "eval", "reference_metrics.json")
    ref = None
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as f:
            ref = json.load(f)
    caps = dict((ref or {}).get("derived_caps") or meta.get("caps") or {})
    task = {"path": path, "name": os.path.basename(path), "task_id": str(meta.get("task_id") or os.path.basename(path)),
            "type": ttype, "metric": meta.get("metric"), "target": target, "caps": caps, "ref": ref}
    task["train"] = pd.read_csv(os.path.join(data, "train.csv"))
    if ttype == "typeI":
        task["test"] = pd.read_csv(os.path.join(data, "test.csv"))
        frames = [task["train"], task["test"]]
    else:
        task["fit"] = pd.read_csv(os.path.join(data, "test_fit.csv"))
        task["test"] = pd.read_csv(os.path.join(data, "test_test.csv"))
        frames = [task["train"], task["fit"], task["test"]]
    if target not in task["train"].columns:
        target = task["train"].columns[0]  # the documented layout: column 0 is the target
        task["target"] = target
    inputs = []
    for col in task["train"].columns:
        if col in (target, "group_id"):
            continue
        if all(col in fr.columns and pd.api.types.is_numeric_dtype(fr[col]) for fr in frames) \
                and not any(fr[col].isna().any() for fr in frames[1:]) and task["train"][col].nunique() > 1:
            inputs.append(col)
    task["inputs"] = inputs
    return task


def _xy(frame, cols, target):
    X = frame[list(cols)].to_numpy(dtype=float) if cols else np.zeros((len(frame), 0))
    y = frame[target].to_numpy(dtype=float)
    return X, y


def _gid(frame) -> np.ndarray:
    return np.array([int(float(v)) for v in frame["group_id"]])


# ---------------------------------------------------------------------------- split profile
def split_profile(task: Dict[str, Any]) -> Dict[str, Any]:
    """Where the test inputs lie relative to the training range, and rows shared with training."""
    import pandas as pd
    cols, target = task["inputs"], task["target"]
    tr = task["train"]
    out: Dict[str, Any] = {}
    if cols:
        lo = tr[cols].min().to_numpy(dtype=float)
        hi = tr[cols].max().to_numpy(dtype=float)
        test_frames = [task["test"]] + ([task["fit"]] if task["type"] == "typeII" else [])
        Xte = np.vstack([fr[cols].to_numpy(dtype=float) for fr in test_frames])
        below, above = Xte < lo, Xte > hi
        out["outside_train_share"] = float(np.mean((below | above).any(axis=1)))
        out["per_input"] = {c: {"below": float(below[:, j].mean()), "above": float(above[:, j].mean())}
                            for j, c in enumerate(cols)}
        share = np.concatenate([below.mean(axis=0), above.mean(axis=0)])
        k = int(np.argmax(share))
        out["direction"] = [cols[k % len(cols)], "below" if k < len(cols) else "above", float(share[k])]
    else:
        out["outside_train_share"] = 0.0
    every = list(cols) + [target]
    th = pd.util.hash_pandas_object(tr[every], index=False).to_numpy()
    scored = task["test"]  # test.csv (Type I) or test_test.csv (Type II): the rows S_N is computed on
    eh = pd.util.hash_pandas_object(scored[every], index=False).to_numpy()
    out["test_rows_in_train"] = int(np.isin(eh, th).sum())
    if cols:
        tx = pd.util.hash_pandas_object(tr[cols], index=False).to_numpy()
        ex = pd.util.hash_pandas_object(scored[cols], index=False).to_numpy()
        out["test_inputs_in_train"] = int(np.isin(ex, tx).sum())
    # the data's own duplication, for comparison: training rows that repeat another training row
    # (Type II: one from another group)
    if task["type"] == "typeII":
        g = _gid(tr)
        pairs = pd.DataFrame({"h": th, "g": g}).drop_duplicates()
        groups_per_row = pairs.groupby("h")["g"].nunique()
        out["train_rows_repeated_across_groups"] = float(np.mean(groups_per_row.reindex(th).to_numpy() > 1))
    else:
        out["train_rows_repeated"] = float(np.mean(pd.Series(th).duplicated(keep=False).to_numpy()))
    out["n_test_rows"] = int(len(eh))  # scored rows
    share = out["outside_train_share"]
    if task["type"] == "typeI":
        out["class"] = "interpolation" if share < 0.05 else ("extrapolation" if share >= 0.5 else "mixed")
    else:
        gtr, gte = set(_gid(tr)), _gid(task["test"])
        out["new_group_share"] = float(np.mean([g not in gtr for g in set(gte)])) if len(gte) else 0.0
        out.update(_within_group_split(task))
        new = "new groups" if out["new_group_share"] >= 0.5 else "shared groups"
        out["class"] = new + (", extrapolating" if out.get("fit_to_test_outside_share", 0) >= 0.5 else "")
    return out


def _within_group_split(task) -> Dict[str, Any]:
    """How each test group is cut into ``test_fit`` and ``test_test`` (inputs only)."""
    cols = task["inputs"]
    gf, gt = _gid(task["fit"]), _gid(task["test"])
    Xf = task["fit"][cols].to_numpy(dtype=float) if cols else np.zeros((len(gf), 0))
    Xt = task["test"][cols].to_numpy(dtype=float) if cols else np.zeros((len(gt), 0))
    fractions, below, above, outside = [], np.zeros(len(cols)), np.zeros(len(cols)), 0
    n_test = 0
    for g in sorted(set(gf) & set(gt)):
        a, b = Xf[gf == g], Xt[gt == g]
        fractions.append(len(a) / (len(a) + len(b)))
        if cols:
            lo, hi = a.min(axis=0), a.max(axis=0)
            below += (b < lo).sum(axis=0)
            above += (b > hi).sum(axis=0)
            outside += int(((b < lo) | (b > hi)).any(axis=1).sum())
        n_test += len(b)
    res: Dict[str, Any] = {"fit_fraction": float(np.median(fractions)) if fractions else 0.5,
                           "fit_to_test_outside_share": outside / n_test if n_test else 0.0, "order": None}
    if cols and n_test:
        share = np.concatenate([below, above]) / n_test
        k = int(np.argmax(share))
        if share[k] >= 0.5:
            res["order"] = [cols[k % len(cols)], "below" if k < len(cols) else "above", float(share[k])]
    return res


# ---------------------------------------------------------------------------- leak audit
def parse_law(path: str) -> Dict[str, Any]:
    """The hidden law's expressions and constants, from the source text (``ast``; nothing runs)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    out: Dict[str, Any] = {"sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(), "n_lines": src.count("\n") + 1}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        out["error"] = f"SyntaxError: {e}"
        return out
    exprs, consts = [], []
    for node in ast.walk(tree):
        value = node.value if isinstance(node, (ast.Return, ast.Assign, ast.AugAssign, ast.AnnAssign)) else None
        if value is not None and any(isinstance(n, ast.BinOp) for n in ast.walk(value)):
            seg = ast.get_source_segment(src, value)
            if seg and len(re.sub(r"\s+", "", seg)) >= 8 and seg not in exprs:
                exprs.append(seg)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool) and float(node.value) not in (0.0, 1.0, -1.0, 2.0):
            consts.append(float(node.value))
    out["expressions"] = exprs
    out["constants"] = sorted(set(consts))
    out["source"] = src
    return out


def answer_key(task_name: str, law: Optional[Dict[str, Any]], ref: Optional[Dict[str, Any]], path: str) -> int:
    """Answer key for the audit: the hidden law's expressions and constants and, when ``ref`` is
    given, the published laws' fitted constants (``reference_metrics.json``)."""
    rows = []
    if law:
        rows += [{"id": f"{task_name}:expression{i}", "group": task_name, "answer": e, "answer_type": "text",
                  "template": "law_expression"} for i, e in enumerate(law.get("expressions", []))]
        rows += [{"id": f"{task_name}:constant{i}", "group": task_name, "answer": c, "answer_type": "number",
                  "template": "law_constant"} for i, c in enumerate(law.get("constants", []))]
    if ref:
        values = sorted({float(v) for b in ref.get("baselines", {}).values() if b.get("kind") == "reference"
                         for v in (b.get("law_constants") or {}).values() if isinstance(v, (int, float))})
        rows += [{"id": f"{task_name}:reference_constant{i}", "group": task_name, "answer": v,
                  "answer_type": "number", "template": "reference_constant"} for i, v in enumerate(values)]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def _decompress(raw: bytes, limit: int) -> Tuple[str, bytes]:
    """joblib's containers (zlib, gzip, bz2, xz/lzma) or a plain pickle; at most ``limit`` bytes."""
    def stream(make, data):
        d = make()
        out = d.decompress(data, limit)
        return out
    attempts = []
    if raw[:2] == b"\x1f\x8b":
        attempts.append(("gzip", lambda: stream(lambda: zlib.decompressobj(31), raw)))
    if raw[:3] == b"BZh":
        attempts.append(("bz2", lambda: stream(bz2.BZ2Decompressor, raw)))
    if raw[:6] == b"\xfd7zXZ\x00" or raw[:1] == b"\x5d":
        attempts.append(("xz/lzma", lambda: stream(lzma.LZMADecompressor, raw)))
    if raw[:1] == b"\x78":
        attempts.append(("zlib", lambda: stream(zlib.decompressobj, raw)))
    if raw[:2] == b"ZF":  # joblib < 0.10: 'ZF' and a length, then zlib
        start = raw.find(b"\x78", 2, 64)
        if start > 0:
            attempts.append(("zlib (old joblib)", lambda: stream(zlib.decompressobj, raw[start:])))
    if raw[:4] == b"\x04\x22\x4d\x18":
        return "lz4 (not inspected: needs the lz4 package)", b""
    for kind, fn in attempts:
        try:
            return kind, fn()
        except Exception:
            continue
    return "none", raw[:limit]


def inspect_joblib(path: str, law: Optional[Dict[str, Any]] = None, limit: int = 1 << 30) -> Dict[str, Any]:
    """What ``state.joblib`` holds, read as pickle opcodes - never unpickled.

    Lists the globals a load would import (``GLOBAL``/``STACK_GLOBAL``) and the strings it stores,
    and checks whether the law's text is among them. joblib writes numpy arrays' bytes into the
    stream after their headers, so the opcode walk stops at the first array; strings after it are
    found by a scan for printable runs."""
    with open(path, "rb") as f:
        raw = f.read()
    kind, data = _decompress(raw, limit)
    out: Dict[str, Any] = {"bytes": len(raw), "compression": kind, "decompressed_bytes": len(data)}
    strings, imports = [], set()
    memo: Dict[int, Any] = {}
    stack: List[Any] = []
    stopped_at = None
    try:
        for op, arg, pos in pickletools.genops(data):
            name = op.name
            if isinstance(arg, str) and name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
                                                 "STRING", "BINSTRING", "SHORT_BINSTRING"):
                strings.append(arg)
                stack.append(arg)
            elif name == "GLOBAL":
                imports.add(str(arg).replace(" ", "."))
                stack.append(None)
            elif name == "STACK_GLOBAL":
                mod, attr = (stack[-2] if len(stack) > 1 else None), (stack[-1] if stack else None)
                imports.add(f"{mod}.{attr}")
                stack.append(None)
            elif name == "MEMOIZE":
                memo[len(memo)] = stack[-1] if stack else None
            elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
                memo[int(arg)] = stack[-1] if stack else None
            elif name in ("BINGET", "LONG_BINGET", "GET"):
                stack.append(memo.get(int(arg)))
            elif name == "STOP":
                stopped_at = pos
                break
            elif name in ("FRAME", "PROTO"):
                pass
            else:
                stack.append(None)
            if len(stack) > 64:
                del stack[:-64]
    except Exception as e:
        stopped_at = f"opcode walk stopped: {type(e).__name__}"
    out["opcode_walk"] = stopped_at
    for m in re.finditer(rb"[\x20-\x7e\t\r\n]{24,}", data):
        strings.append(m.group().decode("ascii"))
    out["imports"] = sorted(imports)
    out["n_strings"] = len(strings)
    out["formula_source_key"] = any(s == "formula_source" for s in strings) or b"formula_source" in data
    compact = [re.sub(r"\s+", "", s) for s in strings if len(s) >= 8]
    if law and law.get("expressions"):
        exprs = [re.sub(r"\s+", "", e) for e in law["expressions"]]
        out["law_expressions_in_state"] = sum(any(e in s for s in compact) for e in exprs)
        out["law_expressions_total"] = len(exprs)
        src = re.sub(r"\s+", "", law.get("source", ""))
        out["formula_py_in_state"] = bool(src) and any(src in s for s in compact)
    return out


CANARY = re.compile(rb"canary", re.I)
GUID = re.compile(rb"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
TEXTLIKE = (".md", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".py", ".csv", ".cff", ".toml", ".html")


def canary_scan(root: str, per_file: int = 8 << 20, per_csv: int = 1 << 20) -> Dict[str, Any]:
    """Canary strings (a marker that lets model trainers drop the files) anywhere in the release."""
    hits, guids, files = [], [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.lower().endswith(TEXTLIKE):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, "rb") as f:
                blob = f.read(per_csv if fn.lower().endswith(".csv") else per_file)
            files += 1
            for m in CANARY.finditer(blob):
                hits.append({"file": os.path.relpath(p, root),
                             "context": blob[max(0, m.start() - 60): m.end() + 80].decode("utf-8", "replace")})
                if len(hits) >= 50:
                    break
            if not fn.lower().endswith(".csv"):
                guids += [(os.path.relpath(p, root), g.decode()) for g in GUID.findall(blob)[:3]]
    return {"files_scanned": files, "canary_hits": hits, "guid_examples": guids[:20], "n_guid_files": len({g[0] for g in guids})}


NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _sig_digits(token: str) -> int:
    mant = re.sub(r"[eE].*$", "", token).lstrip("+-").replace(".", "").lstrip("0")
    return len(mant)


def locate_hits(key_path: str, public: Sequence[str], base: str = "", window: int = 100) -> List[Dict[str, Any]]:
    """Where each answer-key item stands in the public files, so a finding can be judged.

    Text: the file and the line. Numbers (printed with >= 4 significant digits, the audit's rule):
    the file, how often, and how many such matches chance would give - the public numbers printed
    with the same decimals within ``window`` steps of the value, divided by the 2 window + 1 steps.
    Files are given relative to ``base``."""
    with open(key_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    files = []
    for p in public:
        if os.path.isdir(p):
            for dp, dn, fn in os.walk(p):
                dn.sort()
                files += [os.path.join(dp, x) for x in sorted(fn) if x.lower().endswith(TEXTLIKE)]
        elif os.path.exists(p):
            files.append(p)
    hits = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        compact = re.sub(r"\s+", "", text)
        tokens = None
        for it in items:
            a = it["answer"]
            if it["answer_type"] == "text":
                if len(str(a).strip()) < 4 or re.sub(r"\s+", "", str(a)) not in compact:
                    continue
                pattern = r"\s*".join(re.escape(ch) for ch in re.sub(r"\s+", "", str(a)))
                m = re.search(pattern, text)
                line = text[:m.start()].count("\n") + 1 if m else None
                hits.append({"id": it["id"], "template": it["template"], "answer": a,
                             "file": os.path.relpath(path, base) if base else path, "line": line,
                             "context": text[max(0, m.start() - 80): m.end() + 80].replace("\n", " ") if m else None})
                continue
            if tokens is None:
                tokens = NUMBER.findall(text)
            value = float(a)
            exact = [t for t in tokens if _sig_digits(t) >= 4 and float(t) == value]
            if not exact:
                continue
            decimals = len(exact[0].split(".")[1].split("e")[0].split("E")[0]) if "." in exact[0] else 0
            step = 10.0 ** -decimals
            near = sum(1 for t in tokens if ("." in t) == (decimals > 0)
                       and (len(t.split(".")[1]) if "." in t else 0) == decimals
                       and abs(float(t) - value) <= window * step + 1e-12)
            hits.append({"id": it["id"], "template": it["template"], "answer": value,
                         "file": os.path.relpath(path, base) if base else path, "count": len(exact), "printed": exact[0], "chance_expected": near / (2 * window + 1)})
    return hits


def audit_task(task: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    from scidatasim.audit import audit_export
    path, name = task["path"], task["name"]
    res: Dict[str, Any] = {}
    formula = os.path.join(path, "simulator", "formula.py")
    law = parse_law(formula) if os.path.exists(formula) else None
    res["law"] = {k: v for k, v in (law or {}).items() if k != "source"} if law else None
    state = os.path.join(path, "simulator", "state.joblib")
    if os.path.exists(state):
        try:
            res["state_joblib"] = inspect_joblib(state, law)
        except Exception as e:  # unreadable: reported, never loaded
            res["state_joblib"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    if law and "error" in law:
        law = None
    keys = os.path.join(out_dir, "answer_keys")
    os.makedirs(keys, exist_ok=True)
    full = os.path.join(keys, f"{name}.jsonl")
    res["answer_key_items"] = answer_key(name, law, task["ref"], full)
    if res["answer_key_items"]:  # what a solver may read: metadata.yaml and data/
        res["solver_facing"] = audit_export([os.path.join(path, "metadata.yaml"), os.path.join(path, "data")],
                                            full, check_arrays=False).to_dict()
        if res["solver_facing"].get("findings"):
            res["solver_facing_hits"] = locate_hits(full, [os.path.join(path, "metadata.yaml"),
                                                           os.path.join(path, "data")], base=path)
    grader = [p for p in (os.path.join(path, "eval"), os.path.join(path, "simulator")) if os.path.exists(p)]
    if law and grader:  # the positive control: the files the release documents as answer keys
        only_law = os.path.join(keys, f"{name}.law.jsonl")
        if answer_key(name, law, None, only_law):
            res["grader_facing"] = audit_export(grader, only_law, check_arrays=False).to_dict()
    if not law:
        res["note"] = "no parseable simulator/formula.py"
    return res


# ---------------------------------------------------------------------------- fitting
def _powers(d: int, degree: int) -> List[Tuple[int, ...]]:
    ps = [p for p in itertools.product(range(degree + 1), repeat=d) if sum(p) <= degree]
    return sorted(ps, key=lambda p: (sum(p), tuple(-e for e in p)))


def _n_terms(d: int, degree: int) -> int:
    return math.comb(d + degree, degree)


def _standardise(X: np.ndarray, log_in: Sequence[bool]) -> Tuple[List[float], List[float]]:
    """Centre (only when the data sit far from 0; the centre counts as a fitted number) and a
    power-of-ten scale (a unit change, not counted)."""
    center, scale = [], []
    with np.errstate(all="ignore"):
        for j in range(X.shape[1]):
            u = np.log(X[:, j]) if log_in[j] else X[:, j]
            u = u[np.isfinite(u)]
            m, s = (float(np.mean(u)), float(np.std(u))) if u.size else (0.0, 1.0)
            c = float(f"{m:.6g}") if abs(m) > 3 * s else 0.0
            rms = float(np.sqrt(np.mean((u - c) ** 2))) if u.size else 1.0
            center.append(c)
            scale.append(10.0 ** round(math.log10(rms)) if rms > 0 and math.isfinite(rms) else 1.0)
    return center, scale


def _knots(z: np.ndarray, k: int) -> List[float]:
    q = np.unique(np.round(np.quantile(z, np.linspace(0.02, 0.98, k)), 12))
    return [float(v) for v in q] if len(q) >= 3 else []


def n_fitted(spec: Dict[str, Any]) -> int:
    """Fitted numbers written into the submission (global part): coefficients other than 0 and +-1,
    centres, knots."""
    n = 0 if spec["mode"] == "group_poly" else sum(1 for c in spec.get("coef", ()) if c not in (0.0, 1.0, -1.0))
    n += sum(1 for c in spec["center"] if c != 0)
    if spec["kind"] == "spline":
        n += sum(len(k) for k in spec["knots"])
    return n


def fit_global(kind: str, size: int, X: np.ndarray, y: np.ndarray, log_y: bool, log_in: Sequence[bool],
               cols: Sequence[str]) -> Dict[str, Any]:
    """A Type I predictor: constant, polynomial of total degree ``size``, or additive natural cubic
    spline with ``size`` knots per input; least squares on (log) y."""
    t = _target(y, log_y)
    if kind == "const":
        return {"mode": "global", "kind": "poly", "inputs": [], "log_in": [], "center": [], "scale": [],
                "powers": [()], "knots": [], "coef": [float(np.mean(t))], "log_y": log_y, "n_local": 0}
    center, scale = _standardise(X, log_in)
    Z = _z(X, log_in, center, scale)
    powers = _powers(X.shape[1], size) if kind == "poly" else []
    knots = [_knots(Z[:, j], size) for j in range(X.shape[1])] if kind == "spline" else []
    F = _basis(Z, kind, powers, knots, True)
    coef = np.linalg.lstsq(F, t, rcond=None)[0]
    return _unfold_linear({"mode": "global", "kind": kind, "inputs": list(cols), "log_in": list(log_in),
                           "center": center, "scale": scale, "powers": powers, "knots": knots,
                           "coef": [float(c) for c in coef], "log_y": log_y, "n_local": 0})


def predict_global(spec: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    if not spec["inputs"]:
        X = X[:, :0]  # the constant reads no input
    Z = _z(X, spec["log_in"], spec["center"], spec["scale"])
    D = _design(Z, "global", spec["kind"], spec["powers"], spec["knots"], spec["coef"])
    return _output(D @ np.asarray(spec["coef"]), spec["log_y"])


def fit_typeii(mode: str, kind: str, size, X: np.ndarray, y: np.ndarray, gid: np.ndarray, log_y: bool,
               log_in: Sequence[bool], cols: Sequence[str], n_local: int, iters: int = 60) -> Dict[str, Any]:
    """A Type II predictor. ``group_poly``: ``n_local`` polynomial terms refitted per group (``size``:
    the powers). Otherwise a global shape g (polynomial or spline) with a per-group offset (fitted
    by within-group least squares), scale, or both (alternating least squares from that start)."""
    t = _target(y, log_y)
    center, scale = _standardise(X, log_in)
    Z = _z(X, log_in, center, scale)
    spec = {"mode": mode, "kind": "poly" if mode == "group_poly" else kind, "inputs": list(cols),
            "log_in": list(log_in), "center": center, "scale": scale, "powers": [], "knots": [], "coef": [],
            "log_y": log_y, "n_local": n_local}
    if mode == "group_poly":
        spec["powers"] = [tuple(p) for p in size]
        return _unfold_linear(spec)
    spec["powers"] = _powers(X.shape[1], size) if kind == "poly" else []
    spec["knots"] = [_knots(Z[:, j], size) for j in range(X.shape[1])] if kind == "spline" else []
    F = _basis(Z, kind, spec["powers"], spec["knots"], mode == "scale")
    _, g_idx = np.unique(gid, return_inverse=True)
    n_g = np.bincount(g_idx).astype(float)

    def demean(v):
        if v.ndim == 1:
            return v - (np.bincount(g_idx, v) / n_g)[g_idx]
        return np.column_stack([demean(v[:, k]) for k in range(v.shape[1])])
    if mode == "scale":
        beta = np.linalg.lstsq(F, t, rcond=None)[0]
    else:  # within-group least squares: exact for the offset, the start for offset and scale
        beta = np.linalg.lstsq(demean(F), demean(t), rcond=None)[0]
    for _ in range(0 if mode == "offset" else iters):
        g = F @ beta
        if mode == "scale":
            gg = np.bincount(g_idx, g * g)
            a, b = np.zeros_like(n_g), np.where(gg > 0, np.bincount(g_idx, g * t) / np.where(gg > 0, gg, 1), 1.0)
        else:
            mg, mt = np.bincount(g_idx, g) / n_g, np.bincount(g_idx, t) / n_g
            dg, dt = g - mg[g_idx], t - mt[g_idx]
            var = np.bincount(g_idx, dg * dg)
            b = np.where(var > 1e-12, np.bincount(g_idx, dg * dt) / np.where(var > 1e-12, var, 1), 0.0)
            a = mt - b * mg
        new = np.linalg.lstsq(F * b[g_idx][:, None], t - a[g_idx], rcond=None)[0]
        spread = float(np.std(F @ new))
        if spread > 0 and np.isfinite(spread):
            new = new / spread
        done = np.allclose(new, beta, rtol=1e-7, atol=1e-10)
        beta = new
        if done:
            break
    spec["coef"] = [float(c) for c in beta]
    return _unfold_linear(spec)


def _unfold_linear(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Fold centres of linear terms into the intercept (exact), so they are not extra fitted
    numbers; in a shape with a per-group scale, make the leading coefficient 1 (the scale absorbs
    it)."""
    linear = spec["kind"] == "poly" and all(sum(p) <= 1 for p in spec["powers"])
    if linear and any(spec["center"]):
        coef = list(spec["coef"])
        if spec["mode"] in ("global", "scale"):  # intercept first
            for k, p in enumerate(spec["powers"]):
                if sum(p) == 1:
                    j = p.index(1)
                    coef[0] -= coef[k] * spec["center"][j] / spec["scale"][j]
        # offset/affine shapes and per-group polynomials: the per-group intercept absorbs the shift
        spec["coef"] = coef
        spec["center"] = [0.0] * len(spec["center"])
    if spec["mode"] == "scale" and spec["coef"]:
        beta = np.asarray(spec["coef"], dtype=float)
        if abs(beta[0]) > 1e-9 * float(np.max(np.abs(beta))):
            spec["coef"] = [float(c) for c in beta / beta[0]]
    return spec


def local_fit_predict(spec: Dict[str, Any], X_fit, y_fit, X_test) -> np.ndarray:
    """What the harness does with one Type II group: fit() on its fit rows, predict() on the rest."""
    mode = spec["mode"]
    D_fit = _design(_z(X_fit, spec["log_in"], spec["center"], spec["scale"]), mode, spec["kind"], spec["powers"],
                    spec["knots"], spec["coef"])
    c = _fit_local(mode, D_fit, _target(y_fit, spec["log_y"]), spec["n_local"])
    D_test = _design(_z(X_test, spec["log_in"], spec["center"], spec["scale"]), mode, spec["kind"], spec["powers"],
                     spec["knots"], spec["coef"])
    return _output(_apply_local(mode, D_test, c), spec["log_y"])


def _subsample(n: int, cap: int, rng) -> np.ndarray:
    return np.sort(rng.choice(n, cap, replace=False)) if n > cap else np.arange(n)


def _folds_typei(X: np.ndarray, profile: Dict[str, Any], cols: Sequence[str], rng) -> Tuple[str, List]:
    n = len(X)
    direction = profile.get("direction")
    if profile.get("outside_train_share", 0) >= 0.2 and direction and direction[0] in cols:
        j = list(cols).index(direction[0])
        order = np.argsort(X[:, j], kind="stable")
        if direction[1] == "above":
            order = order[::-1]  # the most extreme rows on the side where the test set lies first
        folds = []
        for frac in (0.1, 0.2, 0.3):
            k = max(1, int(round(frac * n)))
            folds.append((np.sort(order[k:]), np.sort(order[:k])))
        return f"edge of {direction[0]} ({direction[1]})", folds
    perm = rng.permutation(n)
    parts = np.array_split(perm, 5)
    return "random 5-fold", [(np.sort(np.concatenate(parts[:i] + parts[i + 1:])), np.sort(parts[i]))
                             for i in range(5)]


def _positive_columns(task) -> List[bool]:
    frames = [task["train"], task["test"]] + ([task["fit"]] if task["type"] == "typeII" else [])
    return [all(bool((fr[c] > 0).all()) for fr in frames) for c in task["inputs"]]


def candidates_typei(task: Dict[str, Any], profile: Dict[str, Any], rng) -> Tuple[str, List[Dict[str, Any]]]:
    """Every variant, with its cross-validated loss on the training data."""
    cols, name = task["inputs"], task["metric"]
    tr = task["train"].dropna(subset=list(cols) + [task["target"]])
    X, y = _xy(tr, cols, task["target"])
    idx = _subsample(len(X), MAX_CV_ROWS, rng)
    Xc, yc = X[idx], y[idx]
    scheme, folds = _folds_typei(Xc, profile, cols, rng)
    pos = _positive_columns(task)
    log_ys = [False, True] if np.all(y > 0) else [False]
    log_xs = [False, True] if any(pos) else [False]
    d = len(cols)
    limit = min(200, len(Xc) // 5)
    variants = [("const", 0, ly, False, []) for ly in log_ys]  # (kind, size, log y, log x, inputs)
    for ly, lx in itertools.product(log_ys, log_xs):
        for deg in range(1, 7):
            if _n_terms(d, deg) > limit:
                break
            variants.append(("poly", deg, ly, lx, list(cols)))
        for k in (3, 4, 5, 6, 8):
            if 1 + d * (k - 1) > limit:
                break
            variants.append(("spline", k, ly, lx, list(cols)))
        if d > 1:  # a curve in one input: the fewest numbers
            for j, col in enumerate(cols):
                if lx and not pos[j]:
                    continue
                variants += [("poly", deg, ly, lx, [col]) for deg in range(1, 5) if deg + 1 <= limit]
    out = []
    fidx = _subsample(len(X), MAX_FIT_ROWS, rng)
    for kind, size, ly, lx, use in variants:
        jj = [list(cols).index(c) for c in use]
        log_in = [lx and pos[j] for j in jj]
        losses = []
        for a, b in folds:
            try:
                spec = fit_global(kind, size, Xc[a][:, jj], yc[a], ly, log_in, use)
                losses.append(_loss(name, metric(name, yc[b], predict_global(spec, Xc[b][:, jj]))))
            except Exception:
                losses.append(math.inf)
        cv = float(np.mean(losses))
        if not np.isfinite(cv):
            continue
        spec = fit_global(kind, size, X[fidx][:, jj], y[fidx], ly, log_in, use)
        one = len(use) == 1 and d > 1
        out.append({"family": _family(kind, "global", ly, lx, one),
                    "variant": _variant(kind, size) + (f" in {use[0]}" if one else ""), "cv_loss": cv,
                    "n_global": n_fitted(spec), "n_local": 0, "spec": spec})
    return scheme, out


def _family(kind: str, mode: str, ly: bool, lx: bool, one: bool = False) -> str:
    base = {("const", "global"): "constant", ("poly", "global"): "polynomial", ("spline", "global"): "spline",
            ("poly", "group_poly"): "per-group polynomial"}.get((kind, mode))
    if base is None:
        base = f"{'polynomial' if kind == 'poly' else 'spline'} shape + per-group " + \
               {"offset": "offset", "scale": "scale", "affine": "offset and scale"}[mode]
    if one and mode != "group_poly":
        base = base.replace(" shape", " shape in one input", 1) if " shape" in base \
            else base.replace("polynomial", "polynomial in one input", 1)
    tags = [t for t, on in (("log y", ly), ("log x", lx)) if on]
    return base + (f" ({', '.join(tags)})" if tags else "")


def _variant(kind: str, size, names: Sequence[str] = ()) -> str:
    if kind == "const":
        return "mean"
    if isinstance(size, int):
        return f"degree {size}" if kind == "poly" else f"{size} knots per input"
    terms = []
    for p in size:
        if not any(p):
            terms.append("1")
            continue
        terms.append("*".join((names[j] if j < len(names) else f"x{j}") + (f"^{e}" if e > 1 else "")
                              for j, e in enumerate(p) if e))
    return "terms " + ", ".join(terms)


def _emulated_split(n: int, X: np.ndarray, split: Dict[str, Any], cols: Sequence[str], rng):
    k = min(n - 1, max(1, int(round(split.get("fit_fraction", 0.5) * n))))
    order = split.get("order")
    if order and order[0] in cols:
        idx = np.argsort(X[:, list(cols).index(order[0])], kind="stable")
        if order[1] == "below":
            idx = idx[::-1]  # test rows lie below the fit rows: fit on the high end
    else:
        idx = rng.permutation(n)
    return np.sort(idx[:k]), np.sort(idx[k:])


def candidates_typeii(task: Dict[str, Any], profile: Dict[str, Any], rng) -> Tuple[str, List[Dict[str, Any]]]:
    cols, name, target = task["inputs"], task["metric"], task["target"]
    m = int(task["caps"].get("max_local_params") or 0)
    tr = task["train"].dropna(subset=list(cols) + [target])
    X, y = _xy(tr, cols, target)
    gid = _gid(tr)
    keep = _subsample(len(X), MAX_FIT_ROWS, rng)
    X, y, gid = X[keep], y[keep], gid[keep]
    counts = dict(zip(*np.unique(gid, return_counts=True)))
    groups = [g for g in counts if counts[g] >= 4]
    groups = list(rng.permutation(groups)[:MAX_CV_GROUPS]) if len(groups) > MAX_CV_GROUPS else groups
    n_folds = min(5, len(groups))
    folds = [groups[i::n_folds] for i in range(n_folds)] if n_folds >= 2 else []
    scheme = (f"held-out groups ({n_folds} folds), each split like the test groups "
              f"({'ordered by ' + profile['order'][0] if profile.get('order') else 'random rows'}, "
              f"fit fraction {profile.get('fit_fraction', 0.5):.2f})")
    pos = _positive_columns(task)
    log_ys = [False, True] if np.all(y > 0) else [False]
    log_xs = [False, True] if any(pos) else [False]
    d = len(cols)
    variants = []  # (mode, kind, size, log_y, log_x, inputs, n_local)
    if m >= 1:
        for ly in log_ys:
            variants.append(("group_poly", "poly", [()], ly, False, [], 1))
            for j, col in enumerate(cols):
                for lx in (log_xs if pos[j] else [False]):
                    for q in range(2, m + 1):
                        variants.append(("group_poly", "poly", [(e,) for e in range(q)], ly, lx, [col], q))
            if 1 + d <= m and d > 1:
                for lx in log_xs:
                    lin = [tuple(int(i == j) for i in range(d)) for j in range(-1, d)]
                    variants.append(("group_poly", "poly", lin, ly, lx, list(cols), 1 + d))
            modes = ["offset", "scale"] + (["affine"] if m >= 2 else [])
            for mode, lx in itertools.product(modes, log_xs):
                n_loc = 2 if mode == "affine" else 1
                for deg in range(1, 4):
                    if _n_terms(d, deg) > 60:
                        break
                    variants.append((mode, "poly", deg, ly, lx, list(cols), n_loc))
                for k in (3, 5):
                    if 1 + d * (k - 1) > 60:
                        break
                    variants.append((mode, "spline", k, ly, lx, list(cols), n_loc))
                if d > 1:  # a shape in one input
                    variants += [(mode, "poly", deg, ly, lx, [col], n_loc) for j, col in enumerate(cols)
                                 if pos[j] or not lx for deg in range(1, 4)]
    out = []
    for mode, kind, size, ly, lx, use, n_local in variants:
        jj = [list(cols).index(c) for c in use]
        log_in = [lx and pos[j] for j in jj]
        losses = []
        for f in folds:
            held = np.isin(gid, f)
            fit_rows = np.flatnonzero(~held)
            if len(fit_rows) > MAX_CV_ROWS:
                fit_rows = np.sort(rng.choice(fit_rows, MAX_CV_ROWS, replace=False))
            try:
                spec = fit_typeii(mode, kind, size, X[fit_rows][:, jj], y[fit_rows], gid[fit_rows], ly, log_in, use,
                                  n_local)
            except Exception:
                losses.append(math.inf)
                continue
            for g in f:
                rows = np.flatnonzero(gid == g)
                a, b = _emulated_split(len(rows), X[rows], profile, cols, rng)
                try:
                    pred = local_fit_predict(spec, X[rows[a]][:, jj], y[rows[a]], X[rows[b]][:, jj])
                    losses.append(_loss(name, metric(name, y[rows[b]], pred)))
                except Exception:
                    losses.append(math.inf)
        cv = float(np.mean(losses)) if losses else math.inf
        if not np.isfinite(cv) and folds:
            continue
        try:
            spec = fit_typeii(mode, kind, size, X[:, jj], y, gid, ly, log_in, use, n_local)
        except Exception:
            continue
        one = len(use) == 1 and d > 1 and mode != "group_poly"
        out.append({"family": _family(kind, mode, ly, lx, one),
                    "variant": _variant(kind, size, use) + (f" in {use[0]}" if one else ""), "cv_loss": cv,
                    "n_global": n_fitted(spec), "n_local": n_local, "spec": spec})
    return scheme, out


def within(c: Dict[str, Any], caps: Dict[str, Any], tier: str) -> bool:
    """Is a candidate inside a limit? The scorer caps ``LOCAL_FITTABLE`` for every tier."""
    if c["n_local"] > int(caps.get("max_local_params") or 0):
        return False
    if tier == "scorer":
        return True
    law = int(caps.get("max_law_constants") or 0)
    return c["n_global"] <= law + (JUDGE_SLACK if tier == "judge" else 0)


# ---------------------------------------------------------------------------- submissions
def _native(v):
    if isinstance(v, (list, tuple)):
        return [_native(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


def _lit(v) -> str:
    """A Python literal (exact floats: repr round-trips)."""
    return repr(_native(v))


def submission_source(spec: Dict[str, Any], task: Dict[str, Any], title: str) -> str:
    typeii = spec["mode"] != "global"
    lines = [f'"""Non-law baseline: {title}.', "",
             "Fitted on train.csv only by scidatasim tools/audit_scilaws.py (not a scientific law).",
             f'Task: {task["task_id"]}; fitted numbers written here: {n_fitted(spec)}; per-group: {spec["n_local"]}."""',
             "import numpy as np", "",
             f"USED_INPUTS = {_lit(spec['inputs'])}",
             "LAW_CONSTANTS = {}", "OTHER_CONSTANTS = {}",
             "LOCAL_FITTABLE = {" + ", ".join(f'"c{k}": {{"init": None}}' for k in range(spec["n_local"])) + "}",
             "",
             f"_MODE = {_lit(spec['mode'])}", f"_KIND = {_lit(spec['kind'])}",
             f"_LOG_IN = {_lit(spec['log_in'])}", f"_CENTER = {_lit(spec['center'])}",
             f"_SCALE = {_lit(spec['scale'])}", f"_POWERS = {_lit([list(p) for p in spec['powers']])}",
             f"_KNOTS = {_lit(spec['knots'])}", f"_COEF = np.array({_lit(spec['coef'])}, dtype=float)",
             f"_LOG_Y = {spec['log_y']!r}", f"_N_LOCAL = {spec['n_local']}", "", "", feature_code(), "", ""]
    if typeii:
        lines += ["def _d(X):",
                  "    return _design(_z(X, _LOG_IN, _CENTER, _SCALE), _MODE, _KIND, _POWERS, _KNOTS, _COEF)", "", "",
                  "def fit(X, y):",
                  "    c = _fit_local(_MODE, _d(X), _target(y, _LOG_Y), _N_LOCAL)",
                  '    return {f"c{k}": float(c[k]) for k in range(_N_LOCAL)}', "", "",
                  "def predict(X, **params):",
                  '    c = np.array([params[f"c{k}"] for k in range(_N_LOCAL)], dtype=float)',
                  "    return _output(_apply_local(_MODE, _d(X), c), _LOG_Y)", ""]
    else:
        lines += ["def predict(X):",
                  "    D = _design(_z(X, _LOG_IN, _CENTER, _SCALE), _MODE, _KIND, _POWERS, _KNOTS, _COEF)",
                  "    return _output(D @ _COEF, _LOG_Y)", ""]
    return "\n".join(lines)


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location(f"_sub_{hashlib.md5(path.encode()).hexdigest()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def local_score(path: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """S_N as the harness computes it, from the submission file (our own code) - the fallback
    without the harness, and the check on its JSON. Also the slowest per-group fit."""
    mod = _load_module(path)
    name, ref = task["metric"], task["ref"]
    best = best_reference(ref, name) if ref else None
    out: Dict[str, Any] = {"best_reference": best}
    used = list(mod.USED_INPUTS)
    if task["type"] == "typeI":
        X, y = _xy(task["test"], used, task["target"])
        yp = np.asarray(mod.predict(X), dtype=float)
        if not np.all(np.isfinite(yp)):
            out.update(s_n=0.0, failed=True)
            return out
        sub = metric(name, y, yp)
        out["raw_metric"] = sub
        if best is None or sub is None:
            out["s_n"] = None
            return out
        rv = ref["baselines"][best]["metrics"][name]
        out["s_n"] = 0.0 if abs(rv - PERFECT.get(name, 0.0)) <= REF_EPS else unit_score(sub, rv, name)
        return out
    gf, gt = _gid(task["fit"]), _gid(task["test"])
    Xf, yf = _xy(task["fit"], used, task["target"])
    Xt, yt = _xy(task["test"], used, task["target"])
    per, slowest = {}, 0.0
    for g in sorted(set(gf) & set(gt)):
        t0 = time.perf_counter()
        try:
            params = mod.fit(Xf[gf == g], yf[gf == g])
            slowest = max(slowest, time.perf_counter() - t0)
            yp = np.asarray(mod.predict(Xt[gt == g], **params), dtype=float)
            per[g] = metric(name, yt[gt == g], yp) if np.all(np.isfinite(yp)) else None
        except Exception:
            per[g] = None
    out["max_fit_seconds"] = slowest
    vals = [v for v in per.values() if v is not None]
    out["raw_metric"] = float(np.mean(vals)) if vals else None
    if best is None:
        out["s_n"] = None
        return out
    scores = []
    for cid, rm in ref["baselines"][best].get("per_cluster", {}).items():
        if not rm or rm.get(name) is None or abs(rm[name] - PERFECT.get(name, 0.0)) <= REF_EPS:
            continue
        sub = per.get(int(cid))
        scores.append(0.0 if sub is None else unit_score(sub, rm[name], name))
    out["s_n"] = float(np.mean(scores)) if scores else 0.0
    return out


def harness_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    for cand in (path, os.path.dirname(os.path.abspath(path))):
        if os.path.exists(os.path.join(cand, "harness", "evaluate_numeric.py")):
            return os.path.abspath(cand)
    raise SystemExit(f"{path}: no harness/evaluate_numeric.py (give the SciLaws-Bench code repository)")


def harness_provenance(root: Optional[str]) -> Dict[str, Any]:
    if not root:
        return {"used": False}
    out: Dict[str, Any] = {"used": True, "path": root, "pinned": PINNED_HARNESS}
    try:
        out["commit"] = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                                       timeout=30).stdout.strip() or None
    except Exception:
        out["commit"] = None
    out["matches_pin"] = out["commit"] == PINNED_HARNESS if out["commit"] else None
    for fn in ("evaluate_numeric.py", "eval_formula.py", "evaluate_validity.py"):
        p = os.path.join(root, "harness", fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[f"sha256 {fn}"] = hashlib.sha256(f.read()).hexdigest()
    out["fit_timeout_enforced"] = hasattr(__import__("signal"), "SIGALRM")
    return out


def run_scorer(root: str, task_dir: str, sub: str, cwd: str, timeout: int = 1800) -> Dict[str, Any]:
    """``evaluate_numeric.py score`` in a child process (isolated mode: no user site, no script
    directory on the path; UTF-8 mode); the JSON it prints, or ``score_error`` with the error."""
    shim = os.path.join(cwd, "_scilaws_scorer.py")
    if not os.path.exists(shim):
        with open(shim, "w", encoding="utf-8") as f:
            f.write(SHIM.lstrip())
    # UTF-8 mode: the harness opens metadata.yaml and the CSVs without an encoding, so on Windows
    # (cp1252) a task whose files hold e.g. "log₁₀" (byte 0x81) crashes before scoring.
    cmd = [sys.executable, "-I", "-X", "utf8", shim, root, "score", os.path.abspath(task_dir), os.path.abspath(sub)]
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "numeric_score": 0.0}
    text = proc.stdout
    start = text.find("{")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else (None, 0)
        if not isinstance(obj, dict):
            raise ValueError("no JSON object")
    except Exception as e:
        return {"status": "score_error", "numeric_score": 0.0, "error": f"{type(e).__name__}: {e}",
                "returncode": proc.returncode, "stderr_tail": proc.stderr[-1500:]}
    keep = ("status", "contract_ok", "violations", "numeric_score", "numeric_score_std", "raw_metric", "error",
            "n_clusters_fitted", "n_clusters_failed", "scoring_interface_version")
    return {k: obj.get(k) for k in keep if k in obj}


# ---------------------------------------------------------------------------- one task
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def fit_and_score(task: Dict[str, Any], profile: Dict[str, Any], out_dir: str, root: Optional[str],
                  jobs: int, rng) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    if not task["metric"]:
        return {"note": "metadata.yaml has no metric"}
    if task["ref"] is None:
        res["note"] = "no eval/reference_metrics.json: S_N cannot be computed"
    builder = candidates_typei if task["type"] == "typeI" else candidates_typeii
    scheme, cands = builder(task, profile, rng)
    res["cv_scheme"] = scheme
    res["n_candidates"] = len(cands)
    if not cands:
        res["note"] = "no candidate could be fitted"
        return res
    chosen: Dict[str, Dict[str, Any]] = {}
    for c in cands:  # the best variant of each family (by cross-validation)
        if within(c, task["caps"], "scorer") and (c["family"] not in chosen or c["cv_loss"] < chosen[c["family"]]["cv_loss"]):
            chosen[c["family"]] = c
    tiers = {}
    for tier in TIERS:
        ok = [c for c in cands if within(c, task["caps"], tier)]
        tiers[tier] = min(ok, key=lambda c: c["cv_loss"]) if ok else None
    sub_dir = os.path.join(out_dir, "submissions", task["name"])
    os.makedirs(sub_dir, exist_ok=True)
    files: Dict[int, str] = {}
    for c in list(chosen.values()) + [t for t in tiers.values() if t is not None]:
        if id(c) in files:
            continue
        title = f"{c['family']}, {c['variant']}"
        source = submission_source(c["spec"], task, title)
        path = os.path.join(sub_dir, f"{_slug(title)[:50]}_{hashlib.sha1(source.encode()).hexdigest()[:6]}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        files[id(c)] = path
    scored: Dict[str, Dict[str, Any]] = {}

    def score(path):
        entry: Dict[str, Any] = {"file": os.path.relpath(path, out_dir)}
        try:
            entry["local"] = local_score(path, task)
        except Exception as e:
            entry["local"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        if root:
            entry["harness"] = run_scorer(root, task["path"], path, out_dir)
        return path, entry

    paths = sorted(set(files.values()))
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for path, entry in pool.map(score, paths):
            scored[path] = entry
    cap = task["caps"].get("fit_timeout_seconds")

    def summary(c):
        e = scored[files[id(c)]]
        h, loc = e.get("harness"), e.get("local", {})
        crashed = bool(h) and h.get("status") == "score_error"  # the harness itself failed: no verdict on the fit
        s_n = loc.get("s_n") if (not h or crashed) else h.get("numeric_score")
        source = "local" if not h else ("local (harness crashed)" if crashed else "harness")
        row = {"family": c["family"], "variant": c["variant"], "cv_loss": c["cv_loss"], "n_global": c["n_global"],
               "n_local": c["n_local"], "s_n": s_n, "s_n_source": source, "s_n_local": loc.get("s_n"),
               "file": e["file"]}
        if h:
            row["harness_status"] = h.get("status")
            row["contract_ok"] = h.get("contract_ok")
            if h.get("violations"):
                row["violations"] = h["violations"]
            if crashed:
                row["harness_error"] = (h.get("error") or "") + " | " + (h.get("stderr_tail") or "")[-800:]
            elif loc.get("s_n") is not None and s_n is not None:
                row["local_agrees"] = abs(loc["s_n"] - s_n) <= 1e-6
        if loc.get("max_fit_seconds") is not None:
            row["max_fit_seconds"] = loc["max_fit_seconds"]
            if cap:
                row["fit_within_timeout"] = loc["max_fit_seconds"] < cap
        return row

    res["families"] = [summary(c) for c in sorted(chosen.values(), key=lambda c: c["cv_loss"])]
    res["tiers"] = {tier: (summary(c) if c is not None else None) for tier, c in tiers.items()}
    return res


def process_task(path: str, out_dir: str, args, root: Optional[str]) -> Dict[str, Any]:
    t0 = time.time()
    task = load_task(path)
    rng = np.random.default_rng(SEED + int(hashlib.md5(task["name"].encode()).hexdigest()[:6], 16))
    res: Dict[str, Any] = {"task": task["name"], "task_id": task["task_id"], "type": task["type"],
                           "metric": task["metric"], "caps": task["caps"], "n_inputs": len(task["inputs"]),
                           "n_train": int(len(task["train"])), "n_test": int(len(task["test"]))}
    profile = split_profile(task)
    if task["type"] == "typeII":
        res["n_test_groups"] = int(len(set(_gid(task["test"]))))
    res["split"] = profile
    if not args.skip_audit:
        res["audit"] = audit_task(task, out_dir)
    if not args.skip_fits:
        res["fits"] = fit_and_score(task, profile, out_dir, root, args.jobs, rng)
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------------------- summary
def agent_scores(root: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not root:
        return {}
    p = os.path.join(root, "baseline_agent", "model_task_scores.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return {tid: {m: float(v) for m, v in (rec.get("real_numeric") or {}).items() if v is not None}
            for tid, rec in d.get("tasks", {}).items()}


def _tier_sn(r: Dict[str, Any], tier: str) -> float:
    t = (r.get("fits") or {}).get("tiers", {}).get(tier)
    return float(t["s_n"]) if t and t.get("s_n") is not None else 0.0


def summarise(results: List[Dict[str, Any]], agents: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    fitted = [r for r in results if r.get("fits") and r["fits"].get("tiers")]
    subsets = {"all": fitted, "typeI": [r for r in fitted if r["type"] == "typeI"],
               "typeII": [r for r in fitted if r["type"] == "typeII"]}
    for cls in sorted({r["split"]["class"] for r in fitted}):
        subsets[f"split: {cls}"] = [r for r in fitted if r["split"]["class"] == cls]
    models = sorted({m for v in agents.values() for m in v})
    out: Dict[str, Any] = {"n_tasks": len(results), "n_fitted": len(fitted), "subsets": {}}
    for label, rows in subsets.items():
        if not rows:
            continue
        entry: Dict[str, Any] = {"n": len(rows)}
        best_agent = [max(agents.get(r["task_id"], {}).values(), default=None) for r in rows]
        for tier in TIERS:
            s = np.array([_tier_sn(r, tier) for r in rows])
            k = int(np.sum(s >= PASS))
            lo, hi = wilson(k, len(s))
            entry[tier] = {"mean_s_n": float(s.mean()), "pass": k, "pass_share": k / len(s), "pass_ci95": [lo, hi],
                           "near": int(np.sum(s >= NEAR)),
                           "eligible": sum(1 for r in rows if r["fits"]["tiers"].get(tier) is not None)}
            pairs = [(a, b) for a, b in zip(s, best_agent) if b is not None]
            if pairs:
                entry[tier]["at_least_best_agent"] = int(sum(a >= b - 1e-9 for a, b in pairs))
                entry[tier]["n_with_agents"] = len(pairs)
        with_agents = [r for r in rows if agents.get(r["task_id"])]
        if models and with_agents:  # tasks the published scores cover; a missing score counts 0, as in theirs
            entry["agents_n"] = len(with_agents)
            entry["agents"] = {m: float(np.mean([agents[r["task_id"]].get(m, 0.0) for r in with_agents]))
                               for m in models}
            entry["non_law_on_agent_tasks"] = {tier: float(np.mean([_tier_sn(r, tier) for r in with_agents]))
                                               for tier in TIERS}
        out["subsets"][label] = entry
    fam: Dict[str, List[float]] = {}
    for r in fitted:
        for f in r["fits"].get("families", []):
            if f.get("s_n") is not None:
                fam.setdefault(f["family"], []).append(float(f["s_n"]))
    out["families"] = {k: {"n": len(v), "pass": int(np.sum(np.array(v) >= PASS)), "mean_s_n": float(np.mean(v))}
                       for k, v in sorted(fam.items(), key=lambda kv: -np.mean(kv[1]))}
    checks = [f for r in fitted for f in r["fits"].get("families", []) if "local_agrees" in f]
    out["local_vs_harness"] = {"compared": len(checks), "agree": sum(bool(f["local_agrees"]) for f in checks)}
    crashed = [(r["task"], f) for r in fitted for f in r["fits"].get("families", []) if "harness_error" in f]
    out["harness_crashes"] = {"submissions": len(crashed), "tasks": sorted({t for t, _ in crashed}),
                              "first_error": crashed[0][1]["harness_error"][-400:] if crashed else None}
    slow = [f for r in fitted for f in r["fits"].get("families", []) if f.get("fit_within_timeout") is False]
    out["fits_over_timeout"] = len(slow)
    return out


def audit_summary(results: List[Dict[str, Any]], canary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    au = [r for r in results if r.get("audit")]
    out: Dict[str, Any] = {"n_tasks": len(au)}
    out["with_formula_py"] = sum(1 for r in au if r["audit"].get("law"))
    solver_high = [r["task"] for r in au if r["audit"].get("solver_facing") and not r["audit"]["solver_facing"]["ok"]]
    out["solver_facing_high"] = solver_high
    out["solver_facing_findings"] = {}
    for r in au:
        for f in (r["audit"].get("solver_facing") or {}).get("findings", []):
            out["solver_facing_findings"].setdefault(f"{f['attack']} ({f['severity']})", []).append(r["task"])
    out["solver_facing_hits"] = {r["task"]: r["audit"]["solver_facing_hits"] for r in au
                                 if r["audit"].get("solver_facing_hits")}
    grader = [r for r in au if r["audit"].get("grader_facing")]
    out["grader_facing_audited"] = len(grader)
    out["grader_facing_answer_lookup"] = sum(
        1 for r in grader if r["audit"]["grader_facing"]["exposure"].get("answer_lookup", {}).get("exposed", 0) > 0)
    states = [r["audit"]["state_joblib"] for r in au if r["audit"].get("state_joblib")]
    out["state_joblib"] = {
        "n": len(states),
        "formula_source_key": sum(1 for s in states if s.get("formula_source_key")),
        "law_text_inside": sum(1 for s in states if s.get("law_expressions_in_state")),
        "formula_py_inside": sum(1 for s in states if s.get("formula_py_in_state")),
        "compression": dict(sorted(_count(s.get("compression", "error") for s in states).items())),
        "imports": dict(sorted(_count(i for s in states for i in s.get("imports", [])).items(),
                               key=lambda kv: -kv[1])[:40]),
    }
    splits = [r["split"] for r in results if r.get("split")]
    out["test_rows_in_train"] = {
        r["task"]: {"rows": r["split"]["test_rows_in_train"], "of": r["split"]["n_test_rows"],
                    "data_repeats": r["split"].get("train_rows_repeated",
                                                   r["split"].get("train_rows_repeated_across_groups"))}
        for r in results if r.get("split", {}).get("test_rows_in_train")}
    out["split_classes"] = dict(_count(s["class"] for s in splits))
    out["canary"] = canary
    return out


def _count(items) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for i in items:
        c[str(i)] = c.get(str(i), 0) + 1
    return c


def _f(v, digits=3) -> str:
    return "-" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def markdown(report: Dict[str, Any]) -> str:
    L = ["# SciLaws-Bench audited with scidatasim's instruments", "",
         f"Generated {report['generated']} by `tools/audit_scilaws.py`. Data: {report['data']}. "
         f"Harness: {report['harness'].get('commit') or 'not used'}"
         + ("" if report["harness"].get("matches_pin") in (None, True) else " (**not the pinned commit**)") + ".", ""]
    if report["harness"].get("used") and not report["harness"].get("fit_timeout_enforced"):
        L += ["The harness ran without its per-group fit timeout (no `signal.alarm` on this system); "
              "the submissions' fit times were measured instead (see below).", ""]
    a = report.get("audit")
    if a:
        L += ["## Leak audit", "",
              f"{a['n_tasks']} tasks; {a['with_formula_py']} ship `simulator/formula.py`. Answer key: its "
              "expressions and constants (the hidden law of the Parallel setting) and the published laws' fitted "
              "constants (`eval/reference_metrics.json`).", "",
              f"* Solver-facing files (`metadata.yaml`, `data/`): high-severity findings in "
              f"{len(a['solver_facing_high'])} tasks" + (f" ({', '.join(a['solver_facing_high'][:10])})"
                                                          if a['solver_facing_high'] else "") + ".",
              ]
        for k, v in a["solver_facing_findings"].items():
            L.append(f"  * {k}: {len(v)} tasks")
        for task, hits in a.get("solver_facing_hits", {}).items():
            for h in hits[:6]:
                where = h["file"]
                if h["template"] == "law_expression" or isinstance(h["answer"], str):
                    L.append(f"    * {task}: `{h['answer']}` ({h['template']}) in {where}, line {h.get('line')}: "
                             f"...{(h.get('context') or '')[:160]}...")
                else:
                    L.append(f"    * {task}: {h['printed']} ({h['template']}) {h['count']}x in {where}; "
                             f"chance would give {h['chance_expected']:.2g}")
        L += [f"* Grader-facing files (`eval/`, `simulator/`), the positive control (the release documents them as "
              f"answer keys): the hidden law found verbatim in {a['grader_facing_answer_lookup']} of "
              f"{a['grader_facing_audited']} tasks (`state.joblib` is binary and left to the next line).",
              f"* `state.joblib` (read as opcodes, never loaded): {a['state_joblib']['n']} files; a "
              f"`formula_source` field in {a['state_joblib']['formula_source_key']}, the law's expressions inside "
              f"{a['state_joblib']['law_text_inside']}, all of `formula.py` inside "
              f"{a['state_joblib']['formula_py_inside']}. Compression: "
              + ", ".join(f"{k} {v}" for k, v in a["state_joblib"]["compression"].items()) + ".",
              "* Globals that unpickling would import (files): "
              + (", ".join(f"`{k}` {v}" for k, v in list(a["state_joblib"]["imports"].items())[:15]) or "none found") + ".",
              f"* Scored test rows identical to a training row (target included): {len(a['test_rows_in_train'])} "
              "tasks" + (". For comparison, the share of training rows that repeat another training row (Type II: "
                         "one of another group) - duplication in the data itself:" if a["test_rows_in_train"] else ".")]
        for k, v in sorted(a["test_rows_in_train"].items(), key=lambda kv: -kv[1]["rows"] / max(1, kv[1]["of"])):
            rep = v.get("data_repeats")
            L.append(f"  * {k}: {v['rows']} of {v['of']} ({v['rows'] / max(1, v['of']):.0%}); in the training data "
                     f"{'-' if rep is None else f'{rep:.0%}'}")
        c = a.get("canary")
        if c:
            L.append(f"* Canary strings: {len(c['canary_hits'])} hits in {c['files_scanned']} text files"
                     + (f" (first: {c['canary_hits'][0]['file']})" if c["canary_hits"] else
                        " - the release carries no marker that would let model trainers exclude it") + ".")
        L.append("")
    s = report.get("summary")
    if s and s["n_fitted"]:
        L += ["## Non-law submissions and S_N", "",
              f"{s['n_fitted']} tasks fitted. A task passes when S_N >= {PASS} (test error at most the best "
              f"published law's); 'near' is S_N >= {NEAR} (within 10 %). Failures count as 0, as in the "
              "benchmark's strict aggregation.", "",
              "| Subset | Tasks | Limit | Eligible | Mean S_N | Pass (95 % CI) | Near | >= best agent |",
              "|---|---:|---|---:|---:|---|---:|---:|"]
        for label, e in s["subsets"].items():
            for tier in TIERS:
                t = e[tier]
                ci = t["pass_ci95"]
                L.append(f"| {label} | {e['n']} | {TIER_TEXT[tier]} | {t['eligible']} | {t['mean_s_n']:.3f} | "
                         f"{t['pass']} ({t['pass_share']:.0%}; {_f(ci[0], 2)}-{_f(ci[1], 2)}) | {t['near']} | "
                         f"{t.get('at_least_best_agent', '-')}/{t.get('n_with_agents', '-')} |")
        allrow = s["subsets"].get("all", {})
        if allrow.get("agents"):
            L += ["", f"Mean S_N on the tasks with published agent scores ({allrow['agents_n']} of {allrow['n']}; "
                  "agents: `baseline_agent/model_task_scores.json`):", "",
                  "| Solver | all | typeI | typeII |", "|---|---:|---:|---:|"]
            for tier in TIERS:
                L.append(f"| non-law, {TIER_TEXT[tier]} | " + " | ".join(
                    _f(s["subsets"].get(k, {}).get("non_law_on_agent_tasks", {}).get(tier))
                    for k in ("all", "typeI", "typeII")) + " |")
            for m in allrow["agents"]:
                L.append(f"| {m} | " + " | ".join(
                    _f(s["subsets"].get(k, {}).get("agents", {}).get(m)) for k in ("all", "typeI", "typeII")) + " |")
        L += ["", "By family (its best variant by cross-validation, where it fits the scorer's limits):", "",
              "| Family | Tasks | Pass | Mean S_N |", "|---|---:|---:|---:|"]
        for k, v in s["families"].items():
            L.append(f"| {k} | {v['n']} | {v['pass']} | {v['mean_s_n']:.3f} |")
        lv, hc = s["local_vs_harness"], s.get("harness_crashes") or {}
        L += ["", f"S_N computed here agrees with the harness on {lv['agree']} of {lv['compared']} submissions both "
              f"scored. Fits slower than the harness's timeout: {s['fits_over_timeout']}."]
        if hc.get("submissions"):
            L += [f"**The harness crashed** on {hc['submissions']} submissions in {len(hc['tasks'])} tasks "
                  f"({', '.join(hc['tasks'][:12])}); S_N computed here is used for them. First error: "
                  f"`{(hc['first_error'] or '').strip()[-300:]}`"]
        L.append("")
    rows = [r for r in report["tasks"] if r.get("fits", {}).get("tiers")]
    if rows:
        L += ["## Per task", "",
              "| Task | Type | Split | Metric | Caps (law/local) | Scorer limit | S_N | Judge limit | S_N | Prompt limit | S_N | Best agent |",
              "|---|---|---|---|---|---|---:|---|---:|---|---:|---:|"]
        agents = report.get("agents", {})
        for r in rows:
            cells = []
            for tier in TIERS:
                t = r["fits"]["tiers"].get(tier)
                cells += [f"{t['family']}, {t['variant']} ({t['n_global']}/{t['n_local']})" if t else "none",
                          _f(t["s_n"]) if t else "-"]
            caps = r["caps"]
            best = max(agents.get(r["task_id"], {}).values(), default=None)
            L.append(f"| {r['task']} | {r['type']} | {r['split']['class']} | {r['metric']} | "
                     f"{caps.get('max_law_constants', '-')}/{caps.get('max_local_params', '-')} | "
                     + " | ".join(cells) + f" | {_f(best)} |")
        L.append("")
    L += ["## Notes", "",
          "* S_N only. The benchmark's validity score S_V (rubrics judged by a code agent, with the anti-hacking "
          "gate) is not computed here; the 'judge' limit applies the gate's counting rule mechanically, and the "
          "judge may still fail a submission it reads as a lookup table.",
          "* Fitted numbers count coefficients, data-derived centres and spline knots; power-of-ten scales and "
          "log transforms are unit changes and are not counted (the gate's rule).",
          "* Families and variants are chosen on the training data only. Test inputs (not targets) decide "
          "only whether a column may be log-transformed (positive everywhere) and how cross-validation folds "
          "imitate the test split.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", required=True, help="the release's tasks/ folder (or one task folder)")
    p.add_argument("--harness", help="the SciLaws-Bench code repository (for evaluate_numeric.py and the agents' scores)")
    p.add_argument("--out", default=os.path.join("experiments_output", "scilaws"))
    p.add_argument("--only", default="", help="comma-separated task folder names")
    p.add_argument("--no-score", action="store_true", help="do not run the harness; S_N computed here")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--skip-fits", action="store_true")
    p.add_argument("--resume", action="store_true", help="keep tasks already in --out/tasks/")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="harness runs in parallel")
    args = p.parse_args(argv)
    root = None if args.no_score else harness_dir(args.harness)
    os.makedirs(os.path.join(args.out, "tasks"), exist_ok=True)
    paths = find_tasks(args.tasks, [s for s in args.only.split(",") if s])
    if not paths:
        raise SystemExit(f"no task folders (metadata.yaml + data/) below {args.tasks}")
    results = []
    for i, path in enumerate(paths, 1):
        cache = os.path.join(args.out, "tasks", os.path.basename(path) + ".json")
        if args.resume and os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                results.append(json.load(f))
            continue
        print(f"[{i}/{len(paths)}] {os.path.basename(path)}", flush=True)
        try:
            r = process_task(path, args.out, args, root)
        except Exception as e:
            r = {"task": os.path.basename(path), "task_id": os.path.basename(path), "error": f"{type(e).__name__}: {e}"[:500]}
            print(f"    error: {r['error']}", flush=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, default=str)
        results.append(r)
    release_root = os.path.dirname(os.path.abspath(args.tasks.rstrip("/\\")))
    canary = None if args.skip_audit else canary_scan(release_root)
    agents = agent_scores(harness_dir(args.harness) if args.harness else None)
    ok = [r for r in results if "error" not in r]
    report = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "data": DATASET,
              "tasks_path": os.path.abspath(args.tasks), "harness": harness_provenance(root),
              "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
              "audit": None if args.skip_audit else audit_summary(ok, canary),
              "summary": None if args.skip_fits else summarise(ok, agents),
              "agents": {r["task_id"]: agents.get(r["task_id"], {}) for r in ok},
              "errors": {r["task"]: r["error"] for r in results if "error" in r}, "tasks": ok}
    with open(os.path.join(args.out, "scilaws_audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    with open(os.path.join(args.out, "scilaws_audit.md"), "w", encoding="utf-8") as f:
        f.write(markdown(report))
    print(f"wrote {os.path.join(args.out, 'scilaws_audit.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
