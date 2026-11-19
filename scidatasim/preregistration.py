"""Pre-registration records: freeze a method, its selection and its predictions before the data
exist, and check them afterwards.

A record is a JSON document. :func:`seal` adds the provenance - git commit and whether the
registered files differ from it, their SHA-256 (with LF line endings, as git stores them, so every
system gives the same values), package versions - and ``record_sha256``, the SHA-256 of the
canonical JSON of everything else (sorted keys, no whitespace). :func:`verify` recomputes both.
The deposit (Zenodo, OSF) is the timestamp; the record only proves what was registered.

Hypotheses may carry a machine-checkable test, so the later check is mechanical:

.. code-block:: yaml

    hypotheses:
      - id: H1
        statement: The false-alarm rate below FAP 0.01 is at most 0.02.
        kind: confirmatory
        metric: false_alarm_rate_0.01
        test: {max: 0.02}                # or {min: x}, {between: [a, b]}, {equals: v, tol: t}
      - id: E1
        kind: exploratory
        statement: ...

:func:`check` takes the outcomes (``{metric: value}``) and reports each hypothesis as passed,
failed, not tested (metric missing) or exploratory (never scored).

Command line: ``sds preregister SPEC --files ... --out record.json`` and
``sds check-preregistration record.json [--outcomes outcomes.json]``.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = "scidatasim.preregistration/1"


def canonical(record: Dict[str, Any]) -> bytes:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=float).encode("utf-8")


def record_hash(record: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical(record)).hexdigest()


def file_hash(path: str) -> str:
    """SHA-256 of a file with CRLF read as LF (git's stored form for text files)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def _git(root: str, *args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def provenance(files: Iterable[str], root: str = ".") -> Dict[str, Any]:
    import numpy
    import scipy
    from . import __version__
    files = sorted(set(files))
    missing = [f for f in files if not os.path.exists(os.path.join(root, f))]
    if missing:
        raise FileNotFoundError(f"registered files not found: {missing}")
    return {"git_commit": _git(root, "rev-parse", "HEAD"),
            "git_dirty": bool(_git(root, "status", "--porcelain", "--", *files)) if files else False,
            "scidatasim": __version__, "python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "line_endings": "LF",
            "files_sha256": {f: file_hash(os.path.join(root, f)) for f in files}}


def seal(record: Dict[str, Any], files: Iterable[str] = (), root: str = ".") -> Dict[str, Any]:
    """A copy of ``record`` with schema, creation time, provenance and ``record_sha256``."""
    out = {k: v for k, v in record.items() if k != "record_sha256"}
    out.setdefault("schema", SCHEMA)
    out.setdefault("created_utc", _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    out["provenance"] = {**(record.get("provenance") or {}), **provenance(files, root)}
    out["record_sha256"] = record_hash(out)
    return out


def verify(record: Dict[str, Any], root: str = ".") -> Dict[str, Any]:
    """``{"hash_ok", "changed_files", "missing_files", "ok"}`` for a record against a checkout."""
    hash_ok = record_hash(record) == record.get("record_sha256")
    files = (record.get("provenance") or {}).get("files_sha256") or {}
    missing = [f for f in files if not os.path.exists(os.path.join(root, f))]
    changed = [f for f, h in files.items() if f not in missing and file_hash(os.path.join(root, f)) != h]
    return {"hash_ok": hash_ok, "changed_files": changed, "missing_files": missing,
            "ok": hash_ok and not changed and not missing}


def _test(test: Dict[str, Any], value: float) -> bool:
    if "max" in test:
        return value <= test["max"]
    if "min" in test:
        return value >= test["min"]
    if "between" in test:
        lo, hi = test["between"]
        return lo <= value <= hi
    if "equals" in test:
        return abs(value - test["equals"]) <= test.get("tol", 0.0)
    raise ValueError(f"unknown test {test}; use max, min, between or equals")


def check(record: Dict[str, Any], outcomes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each hypothesis with its outcome: ``passed`` True/False, or ``None`` when not tested."""
    rows = []
    for h in record.get("hypotheses") or []:
        row = {"id": h.get("id"), "kind": h.get("kind", "confirmatory"), "statement": h.get("statement", "")}
        metric, test = h.get("metric"), h.get("test")
        if row["kind"] == "exploratory" or not metric or not test:
            row.update(status="exploratory" if row["kind"] == "exploratory" else "no machine test", passed=None)
        elif metric not in outcomes or outcomes[metric] is None:
            row.update(status="not tested", passed=None)
        else:
            value = float(outcomes[metric])
            ok = _test(test, value)
            row.update(status="passed" if ok else "failed", passed=ok, metric=metric, value=value, test=test)
        rows.append(row)
    return rows


def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def write(record: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False, default=float)
        f.write("\n")


def markdown(record: Dict[str, Any], verification: Dict[str, Any], rows: Optional[List[Dict[str, Any]]] = None) -> str:
    lines = [f"# {record.get('title', 'Pre-registration')}", "",
             f"- record_sha256: {record.get('record_sha256')} ({'ok' if verification['hash_ok'] else 'MISMATCH'})",
             f"- commit: {(record.get('provenance') or {}).get('git_commit', '')}",
             f"- registered files: {'unchanged' if not verification['changed_files'] else 'CHANGED: ' + ', '.join(verification['changed_files'])}"
             + (f"; missing: {', '.join(verification['missing_files'])}" if verification["missing_files"] else "")]
    if rows is not None:
        lines += ["", "| id | kind | status | value | test |", "|---|---|---|---|---|"]
        lines += [f"| {r['id']} | {r['kind']} | {r['status']} | {r.get('value', '')} | {r.get('test', '')} |" for r in rows]
    return "\n".join(lines) + "\n"
