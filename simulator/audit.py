# simulator/audit.py
"""Leak audit: try to recover hidden answers from the files you are about to publish.

A benchmark built from simulations is only as secret as its public files. This module runs the
known shortcut attacks against a *public* export (a Hugging Face folder, a tasks JSONL file, a
copied bundle directory, anything) using the *private* runs it came from as the answer key, and
reports which attacks would succeed::

    from simulator.audit import audit_export
    report = audit_export("hf_dataset/", private="results/design_001")
    print(report.summary())
    assert report.ok   # no high-severity leak

Attacks (``Finding.attack``):

``private_field``
    A public JSON/JSONL/CSV/Parquet record carries a field that should be private: ``answer``,
    a numeric ``tolerance`` (questions publish ``tolerance_rule`` instead), ``ground_truth``.
``seed_replay``
    A run's seed appears in public files. The simulator and model code are open, so seed plus
    design conditions regenerate the run - answers included.
``config_hash``
    A run's ``config_sha256`` appears in public files. If the configuration space is enumerable
    (a design grid around a public base config), hashing every candidate recovers the hidden
    configuration.
``tolerance_inversion``
    A public numeric tolerance is a fixed fraction of the answer (``5 % of the true value``),
    so ``answer = tolerance / 0.05``. The attack is actually run; findings count questions it
    answers correctly.
``answer_lookup``
    A non-trivial numeric answer, or an expression/text answer, appears verbatim among the
    public numbers or text, e.g. an exact (noise-free) table next to a figure question.
``latent_exposure``
    Most values of a hidden output (role ``latent``, ``parameter`` or ``label``) appear among
    the public numbers: the noise-free ground truth is effectively published.

:func:`audit_environment` runs the analogous checks on an interactive
:class:`simulator.envs.ExperimentEnv`: nothing in the public observation or state may reveal
the episode seed or the hidden truth.

The audit finds *leaks of the known kinds*; passing it is necessary, not sufficient. It does
not judge whether answers can be inferred legitimately from the data (that is the task).
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, os.PathLike]

SEVERITY = ("high", "medium", "info")
TEXT_SUFFIXES = (".json", ".jsonl", ".csv", ".tsv", ".md", ".txt", ".yaml", ".yml", ".dot", ".py", ".toml",
                 ".html", ".svg")
PRIVATE_FIELDS = {"answer": "high", "answers": "high", "ground_truth": "high",
                  "ground_truth_parameters": "high", "tolerance": "medium"}
SEED_FIELDS = ("seed", "seeds", "base_seed", "episode_seed", "law_seed")
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_RULE = re.compile(r"([0-9]*\.?[0-9]+)\s*%\s*of the (?:true |exact )?(?:value|answer)", re.I)


@dataclass
class Finding:
    attack: str
    severity: str
    detail: str
    file: Optional[str] = None
    count: int = 1


@dataclass
class AuditReport:
    findings: List[Finding] = field(default_factory=list)
    #: ``{attack: {"exposed": k, "total": n}}`` for the attacks that were measured.
    exposure: Dict[str, Dict[str, int]] = field(default_factory=dict)
    files_scanned: int = 0

    @property
    def ok(self) -> bool:
        """``True`` if no high-severity finding."""
        return not any(f.severity == "high" for f in self.findings)

    def by_attack(self, attack: str) -> List[Finding]:
        return [f for f in self.findings if f.attack == attack]

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "files_scanned": self.files_scanned, "exposure": self.exposure,
                "findings": [asdict(f) for f in self.findings]}

    def summary(self) -> str:
        lines = [f"Leak audit: {'PASS' if self.ok else 'FAIL'} ({self.files_scanned} public files scanned)"]
        for attack, e in self.exposure.items():
            lines.append(f"  {attack:<20} {e['exposed']}/{e['total']} exposed")
        for f in sorted(self.findings, key=lambda f: SEVERITY.index(f.severity)):
            where = f" [{f.file}]" if f.file else ""
            lines.append(f"  {f.severity.upper():<6} {f.attack}: {f.detail}{where}")
        if not self.findings:
            lines.append("  no findings")
        return "\n".join(lines)


# ---------------------------------------------------------------------------- public side
@dataclass
class _Public:
    numbers: List[float] = field(default_factory=list)
    #: Numbers printed in text with few digits: (value, significant digits), e.g. 13.71 -> 4.
    printed: List[Tuple[float, int]] = field(default_factory=list)
    integers: set = field(default_factory=set)
    seed_values: Dict[int, str] = field(default_factory=dict)  # value -> file, from seed-named fields
    seed_floats: Dict[float, str] = field(default_factory=dict)  # seeds written as floats (1.69e18)
    text: List[Tuple[str, str]] = field(default_factory=list)  # (file, text)
    records: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)  # question-like records
    question_numbers: List[float] = field(default_factory=list)  # numbers quoted in public questions
    fields: List[Tuple[str, str, str]] = field(default_factory=list)  # (file, key, kind)
    design_point_files: List[str] = field(default_factory=list)
    unreadable: List[Tuple[str, str]] = field(default_factory=list)  # (file, reason)
    files: int = 0

    def add_number(self, value: Any) -> None:
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            self.integers.add(int(value))  # exact: seeds exceed float precision
        elif isinstance(value, str) and value.lstrip("+-").isdigit():
            self.integers.add(int(value))
        try:
            v = float(value)
        except (TypeError, ValueError, OverflowError):
            return
        if math.isfinite(v):
            self.numbers.append(v)
            if v == int(v) and abs(v) < 2 ** 53:
                self.integers.add(int(v))

    def add_text(self, path: str, text: str) -> None:
        self.text.append((path, text))
        for m in _NUMBER.finditer(text):
            token = m.group()
            self.add_number(token)
            digits = _printed_digits(token)
            if digits >= 4:
                try:
                    self.printed.append((float(token), digits))
                except (ValueError, OverflowError):
                    pass

    def add_seed(self, value: Any, path: str) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, str):
            value = value.strip()
            if value.lstrip("+-").isdigit():
                self.seed_values.setdefault(int(value), path)
                return
            try:
                value = float(value)
            except ValueError:
                return
        if isinstance(value, (int, np.integer)):
            self.seed_values.setdefault(int(value), path)
        elif isinstance(value, (float, np.floating)) and math.isfinite(value) and float(value) == int(value):
            self.seed_floats.setdefault(float(value), path)
            self.seed_values.setdefault(int(value), path)


def _printed_digits(token: str) -> int:
    mantissa = token.lstrip("+-").lower().split("e")[0]
    digits = mantissa.replace(".", "").lstrip("0")
    return len(digits) if "." in mantissa else len(digits.rstrip("0"))


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, list, tuple, dict)):
        return len(value) == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    return False


def _iter_files(paths: Union[PathLike, Sequence[PathLike]]) -> Iterator[str]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    for p in paths:
        p = os.fspath(p)
        if os.path.isfile(p):
            yield p
            continue
        if not os.path.isdir(p):
            raise FileNotFoundError(f"public path '{p}' does not exist")
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
            for name in sorted(files):
                yield os.path.join(root, name)


def _is_seed_key(low: str) -> bool:
    return low in SEED_FIELDS or low.endswith("_seed") or low.endswith("_seeds")


def _walk_json(obj: Any, pub: _Public, path: str, parent_key: str = "") -> None:
    if isinstance(obj, dict):
        if "question" in obj or "prompt" in obj:
            pub.records.append((path, obj))
            for key in ("question", "prompt"):
                if isinstance(obj.get(key), str):
                    pub.question_numbers.extend(float(m.group()) for m in _NUMBER.finditer(obj[key])
                                                if _finite_float(m.group()))
        for k, v in obj.items():
            key = str(k)
            low = key.lower()
            if low in PRIVATE_FIELDS and not _empty(v):
                if low != "tolerance" or (isinstance(v, (int, float, np.number)) and not isinstance(v, bool)):
                    pub.fields.append((path, key, PRIVATE_FIELDS[low]))
            if low == "design_point" and v is not None:
                pub.design_point_files.append(path)
            if _is_seed_key(low):
                for s in (v if isinstance(v, (list, tuple)) else [v]):
                    pub.add_seed(s, path)
            if low == "config_sha256" and isinstance(v, str):
                pub.text.append((path, v))
            _walk_json(v, pub, path, key)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_json(v, pub, path, parent_key)
    elif isinstance(obj, np.ndarray):
        _walk_json(obj.tolist(), pub, path, parent_key)
    elif isinstance(obj, (bool, np.bool_)) or obj is None:
        return
    elif isinstance(obj, (int, float, np.number)):
        pub.add_number(obj)
    elif isinstance(obj, str):
        stripped = obj.strip()
        if stripped[:1] in "[{":  # JSON text inside a JSON/Parquet value
            try:
                _walk_json(json.loads(stripped), pub, path, parent_key)
                return
            except ValueError:
                pass
        pub.add_text(path, obj)


def _finite_float(token: str) -> bool:
    try:
        return math.isfinite(float(token))
    except (ValueError, OverflowError):
        return False


def _numeric(series) -> bool:
    from pandas.api.types import is_bool_dtype, is_numeric_dtype
    return is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)


def _read_table(path: str, pub: _Public, frame) -> None:
    for col in frame.columns:
        low = str(col).lower()
        series = frame[col]
        if low in PRIVATE_FIELDS:
            values = [v for v in series.tolist() if not _empty(v) and not (isinstance(v, float) and math.isnan(v))]
            if values and (low != "tolerance" or _numeric(series)):
                pub.fields.append((path, str(col), PRIVATE_FIELDS[low]))
        if low == "design_point" and series.notna().any():
            pub.design_point_files.append(path)
        is_seed = _is_seed_key(low)
        if _numeric(series):
            for v in series.dropna().tolist():  # Python ints stay exact
                pub.add_number(v)
                if is_seed:
                    pub.add_seed(v, path)
            continue
        for v in series.tolist():
            if isinstance(v, (bytes, bytearray, memoryview)) or v is None:
                continue
            if isinstance(v, dict) and "bytes" in v:  # datasets.Image struct
                continue
            if isinstance(v, float) and not math.isfinite(v):
                continue
            if is_seed:
                pub.add_seed(v, path)
            _walk_json(v, pub, path, str(col))
    if {"question", "prompt"} & set(map(str, frame.columns)):
        for row in frame.to_dict("records"):
            pub.records.append((path, row))


def _read_hdf5(path: str, pub: _Public) -> None:
    import h5py  # ImportError -> reported as unreadable

    def attrs(obj):
        for key, value in obj.attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            _walk_json({key: value.tolist() if isinstance(value, np.ndarray) else
                        value.item() if isinstance(value, np.generic) else value}, pub, path)

    def visit(_name, obj):
        attrs(obj)
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            if isinstance(data, bytes):
                pub.add_text(path, data.decode("utf-8", "replace"))
            elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
                for v in np.ravel(data):
                    pub.add_number(v)
            elif isinstance(data, (np.number, int, float)):
                pub.add_number(data)

    with h5py.File(path, "r") as f:
        attrs(f)
        f.visititems(visit)


def _scan_public(paths) -> _Public:
    import pandas as pd
    pub = _Public()
    for path in _iter_files(paths):
        suffix = os.path.splitext(path)[1].lower()
        if suffix not in TEXT_SUFFIXES + (".parquet", ".npy", ".npz", ".h5", ".hdf5"):
            continue  # images and other binary files carry no answers in machine-readable form
        try:
            if suffix == ".parquet":
                _read_table(path, pub, pd.read_parquet(path))
            elif suffix in (".npy", ".npz"):
                loaded = np.load(path, allow_pickle=False)
                arrays = [loaded[k] for k in loaded.files] if suffix == ".npz" else [loaded]
                for arr in arrays:
                    if np.issubdtype(arr.dtype, np.number):
                        for v in np.ravel(arr):
                            pub.add_number(v)
            elif suffix in (".h5", ".hdf5"):
                _read_hdf5(path, pub)
            elif suffix in (".csv", ".tsv"):
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                pub.add_text(path, text)
                try:
                    frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=str)
                except Exception:  # not tabular after all: the text scan above still counts
                    frame = None
                if frame is not None:
                    for col in frame.columns:
                        low = str(col).lower()
                        if low in PRIVATE_FIELDS and frame[col].notna().any():
                            if low != "tolerance" or any(_finite_float(v) for v in frame[col].dropna()):
                                pub.fields.append((path, str(col), PRIVATE_FIELDS[low]))
                        if low == "design_point" and frame[col].notna().any():
                            pub.design_point_files.append(path)
                        if _is_seed_key(low):
                            for v in frame[col].dropna():
                                pub.add_seed(v, path)
            elif suffix in (".json", ".jsonl"):
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                pub.text.append((path, text))
                if suffix == ".jsonl":
                    bad = 0
                    for line in text.splitlines():
                        if line.strip():
                            try:
                                _walk_json(json.loads(line), pub, path)
                            except ValueError:
                                bad += 1
                    if bad:
                        pub.unreadable.append((path, f"{bad} malformed JSON line(s)"))
                        pub.add_text(path, text)  # still search the raw text
                else:
                    try:
                        _walk_json(json.loads(text), pub, path)
                    except ValueError:
                        pub.unreadable.append((path, "malformed JSON"))
                        pub.add_text(path, text)
            else:
                with open(path, encoding="utf-8", errors="replace") as f:
                    pub.add_text(path, f.read())
        except RecursionError:
            pub.unreadable.append((path, "nested too deeply to scan"))
            continue
        except Exception as e:  # unreadable file: it cannot be cleared, so it is reported
            pub.unreadable.append((path, f"{type(e).__name__}: {e}"[:200]))
            continue
        pub.files += 1
    return pub


# ---------------------------------------------------------------------------- private side
@dataclass
class _Private:
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    seeds: Dict[str, int] = field(default_factory=dict)  # experiment_id -> seed
    hashes: Dict[str, str] = field(default_factory=dict)  # experiment_id -> config_sha256
    hidden_arrays: List[Tuple[str, str, np.ndarray]] = field(default_factory=list)  # (run, name, values)
    #: Sorted values of each run's coordinate outputs: an answer on the public grid (an argmax,
    #: a crossing time) appears there by construction and is not a leak.
    grids: Dict[str, np.ndarray] = field(default_factory=dict)
    design_columns: List[str] = field(default_factory=list)  # parameter columns of design tables


def _load_private(private, include_arrays: bool = True) -> _Private:
    from .export.tasks import _run_dirs, collect_bundle_tasks
    from .persistence import load_experiment_record
    from .utils import GROUND_TRUTH_ROLES, Role
    priv = _Private()
    for source in ([private] if isinstance(private, (str, os.PathLike)) else list(private)):
        design_csv = os.path.join(os.fspath(source), "design.csv")
        if os.path.exists(design_csv):
            import pandas as pd
            priv.design_columns += [c for c in pd.read_csv(design_csv, nrows=0).columns
                                    if c not in ("design_point", "replicate", "seed")]
    run_dirs = _run_dirs(private)
    priv.tasks = collect_bundle_tasks(run_dirs)
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "experiment_record.json"), encoding="utf-8") as f:
            rec = json.load(f)
        eid = rec["experiment_id"]
        seed = (rec.get("randomness") or {}).get("seed", rec.get("config", {}).get("seed"))
        if seed is not None:
            priv.seeds[eid] = int(seed)
        h = (rec.get("provenance") or {}).get("config_sha256")
        if h:
            priv.hashes[eid] = h
        loaded = load_experiment_record(os.path.dirname(run_dir), eid, verify=False)
        grid = []
        for name, info in loaded.output_data.items():
            data, desc = info["data"], info["descriptor"]
            if desc.role is Role.COORDINATE and isinstance(data, np.ndarray) \
                    and np.issubdtype(data.dtype, np.number):
                grid.append(np.ravel(data).astype(float))
            if include_arrays:
                if desc.role in GROUND_TRUTH_ROLES and isinstance(data, np.ndarray) \
                        and np.issubdtype(data.dtype, np.number) and data.size >= 5:
                    priv.hidden_arrays.append((eid, name, np.ravel(data).astype(float)))
        priv.grids[eid] = np.unique(np.concatenate(grid)) if grid else np.empty(0)
    return priv


# ---------------------------------------------------------------------------- helpers
def _significant_digits(value: float) -> int:
    mantissa = f"{abs(value):.15g}".split("e")[0]
    return max(1, len(mantissa.replace(".", "").strip("0")))


def _nontrivial(value: float) -> bool:
    """Numbers specific enough that a coincidental match is unlikely."""
    if not math.isfinite(value) or value == 0:
        return False
    if value == int(value):
        return abs(value) >= 10 ** 5
    return _significant_digits(value) >= 4


def _contains(sorted_numbers: np.ndarray, value: float, rel_tol: float) -> bool:
    if sorted_numbers.size == 0:
        return False
    tol = rel_tol * max(abs(value), 1e-300)
    i = np.searchsorted(sorted_numbers, value - tol)
    return i < sorted_numbers.size and sorted_numbers[i] <= value + tol


def _printed(printed: List[Tuple[float, int]], value: float) -> bool:
    """``value`` printed to the digits of a public number (``13.7094`` for 13.70937...)."""
    if not printed or not math.isfinite(value) or value == 0:
        return False
    import bisect
    lo = bisect.bisect_left(printed, (value - abs(value) * 1e-3, -1))
    for number, digits in printed[lo:]:
        if number > value + abs(value) * 1e-3:
            break
        if float(f"{value:.{digits - 1}e}") == number:
            return True
    return False


def _short(path: Optional[str], root: Optional[str]) -> Optional[str]:
    if path is None or root is None:
        return path
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


# ---------------------------------------------------------------------------- main entry
def audit_export(public: Union[PathLike, Sequence[PathLike]], private: Union[PathLike, Sequence[PathLike]],
                 rel_tol: float = 1e-6, latent_fraction: float = 0.9,
                 check_arrays: bool = True, ignore: Sequence[str] = ()) -> AuditReport:
    """Attack a public export with the private runs as the answer key.

    Args:
        public: Folder(s) or file(s) that will be published (all supported files below a folder
            are read: JSON, JSONL, CSV, Parquet, ``.npy``/``.npz``, HDF5 (needs h5py), Markdown and
            other text). A supported file that cannot be read is a high-severity finding: it cannot
            be cleared.
        private: Run directory, design directory, or a list of either (the source of the export).
        rel_tol: Relative tolerance for a public number to count as a copy of a hidden value.
        latent_fraction: Fraction of a hidden array's values that must appear publicly to report
            ``latent_exposure``.
        check_arrays: Load hidden arrays for ``latent_exposure`` (slower on big designs).
        ignore: Output names to leave out of ``latent_exposure`` (outputs published on purpose).
            Hidden arrays whose values are quoted in the public questions are left out anyway.

    Known blind spots: seeds below 100000 are only recognised under seed-named fields, not in
    free text; answers are found when copied exactly, printed to at least 4 significant digits,
    or (formulas and text) with different spacing - not when transformed otherwise.
    """
    pub = _scan_public(public)
    priv = _load_private(private, include_arrays=check_arrays)
    root = os.fspath(public) if isinstance(public, (str, os.PathLike)) and os.path.isdir(public) else None
    report = AuditReport(files_scanned=pub.files)
    numbers = np.unique(np.asarray(pub.numbers, dtype=float))
    alltext = "\n".join(t for _, t in pub.text)
    compact = re.sub(r"\s+", "", alltext)

    for path, reason in pub.unreadable:
        report.findings.append(Finding("unreadable", "high", f"could not scan this file ({reason}); "
                                       "it cannot be cleared", _short(path, root)))
    if pub.files == 0:
        report.findings.append(Finding("nothing_scanned", "high", "no supported public file was read; "
                                       "check the public path(s)"))

    # private fields --------------------------------------------------------
    seen = set()
    for path, key, severity in pub.fields:
        if (path, key) in seen:
            continue
        seen.add((path, key))
        n = sum(1 for p, k, _ in pub.fields if p == path and k == key)
        report.findings.append(Finding("private_field", severity,
                                       f"field '{key}' is published ({n} occurrence(s))", _short(path, root), n))

    # seeds -----------------------------------------------------------------
    exposed = []
    for eid, seed in priv.seeds.items():
        where = pub.seed_values.get(seed) or pub.seed_floats.get(float(seed))
        if where is None and abs(seed) >= 10 ** 5 and seed in pub.integers:
            where = next((p for p, t in pub.text if str(seed) in t), "(numeric field)")
        if where is not None:
            exposed.append((eid, where))
    report.exposure["seed_replay"] = {"exposed": len(exposed), "total": len(priv.seeds)}
    if exposed:
        report.findings.append(Finding(
            "seed_replay", "high", f"{len(exposed)}/{len(priv.seeds)} run seeds are public; with the open "
            f"simulator each such run (and its answers) can be regenerated, e.g. {exposed[0][0]}",
            _short(exposed[0][1], root), len(exposed)))

    # config hashes ---------------------------------------------------------
    hits = [eid for eid, h in priv.hashes.items() if h in alltext]
    report.exposure["config_hash"] = {"exposed": len(hits), "total": len(priv.hashes)}
    if hits:
        report.findings.append(Finding(
            "config_hash", "medium", f"{len(hits)}/{len(priv.hashes)} config_sha256 values are public; if the "
            "configuration space is enumerable (design grid + base config) hashing candidates recovers the "
            "hidden configuration", None, len(hits)))

    # tolerance inversion ---------------------------------------------------
    from .bundle import check_answer
    answers = {t["id"]: t for t in priv.tasks}
    by_question = {}
    for t in priv.tasks:
        by_question.setdefault((t["experiment_id"], t["question"]), t)
    candidates, solved = 0, 0
    for path, rec in pub.records:
        tol = rec.get("tolerance")
        if not isinstance(tol, (int, float)) or isinstance(tol, bool) or not tol:
            continue
        task = answers.get(rec.get("id")) or by_question.get((rec.get("experiment_id"), rec.get("question")))
        if task is None or task["answer_type"] != "number":
            continue
        candidates += 1
        m = _RULE.search(str(rec.get("tolerance_rule") or task.get("tolerance_rule") or ""))
        if not m:
            continue
        guess = float(tol) / (float(m.group(1)) / 100.0)
        if any(check_answer(task, g) for g in (guess, -guess)):
            solved += 1
    if candidates:
        report.exposure["tolerance_inversion"] = {"exposed": solved, "total": candidates}
        if solved:
            report.findings.append(Finding(
                "tolerance_inversion", "high", f"{solved}/{candidates} public numeric tolerances invert to a "
                "correct answer (tolerance / rule fraction)", None, solved))

    # design point --------------------------------------------------------
    if pub.design_point_files and priv.design_columns:
        report.findings.append(Finding(
            "design_point", "medium", "design-point indices are public; with the design table (public, or "
            "regenerable from code and seed) they map runs to their design conditions "
            f"({', '.join(priv.design_columns)})", _short(pub.design_point_files[0], root)))

    # answer lookup ---------------------------------------------------------
    printed = sorted(pub.printed)
    lookups, total = [], 0
    for t in priv.tasks:
        a = t.get("answer")
        if t["answer_type"] == "choice":
            continue  # the answer is among the public choices by design
        if t["answer_type"] in ("number", "integer") and isinstance(a, (int, float)) and not isinstance(a, bool):
            if not _nontrivial(float(a)) or _contains(priv.grids.get(t["experiment_id"], np.empty(0)),
                                                      float(a), rel_tol):
                continue
            total += 1
            if _contains(numbers, float(a), rel_tol) or _printed(printed, float(a)):
                lookups.append(t)
        elif isinstance(a, str) and len(a.strip()) >= 4:
            total += 1
            if a.strip() in alltext or re.sub(r"\s+", "", a) in compact:
                lookups.append(t)
    report.exposure["answer_lookup"] = {"exposed": len(lookups), "total": total}
    if lookups:
        templates = sorted({str(t.get("template")) for t in lookups})
        report.findings.append(Finding(
            "answer_lookup", "high", f"{len(lookups)}/{total} non-trivial answers appear verbatim in public files "
            f"(templates: {', '.join(templates)}), e.g. {lookups[0]['id']}", None, len(lookups)))

    # latent exposure -------------------------------------------------------
    if check_arrays:
        leaked = []
        quoted = np.unique(np.asarray(pub.question_numbers, dtype=float))
        for eid, name, values in priv.hidden_arrays:
            if name in ignore:
                continue
            specific = values[[_nontrivial(v) for v in values]]
            if specific.size < 5:
                continue
            if np.mean([_contains(quoted, v, rel_tol) for v in specific]) >= latent_fraction:
                continue  # quoted in the questions on purpose (e.g. given coordinates)
            frac = np.mean([_contains(numbers, v, rel_tol) for v in specific])
            if frac >= latent_fraction:
                leaked.append(f"{eid}:{name}")
        report.exposure["latent_exposure"] = {"exposed": len(leaked), "total": len(priv.hidden_arrays)}
        if leaked:
            report.findings.append(Finding(
                "latent_exposure", "high", f"{len(leaked)} hidden output arrays appear in public files "
                f"(e.g. {leaked[0]}): the noise-free ground truth is published", None, len(leaked)))
    return report


def _seed_values(obj: Any, under_seed: bool = False) -> Iterator[int]:
    """Integers stored under seed-named keys anywhere in ``obj``."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _seed_values(v, under_seed or "seed" in str(k).lower())
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _seed_values(v, under_seed)
    elif under_seed and isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        yield int(obj)


def audit_environment(env, seeds: Iterable[int] = (0, 1, 2), probe: Optional[Dict[str, Any]] = None
                      ) -> AuditReport:
    """Check that an :class:`~simulator.envs.ExperimentEnv` hides its seed and its truth.

    For each episode seed: the public reset observation, the public ``state``, and (if ``probe``,
    an experiment action, is given) the observation after one experiment are serialised and
    searched for the seed, the derived seeds and the hidden truth values.
    """
    report = AuditReport()
    seeds = list(seeds)
    seed_hits, truth_hits, truth_total = 0, 0, 0
    for seed in seeds:
        public = [env.reset(seed=seed), env.state]
        if probe is not None:
            public.append(env.step(probe))
            public.append(env.state)
        pub = _Public()
        for obj in public:
            _walk_json(json.loads(json.dumps(obj, default=str)), pub, f"episode {seed}")
        text = json.dumps(public, default=str)
        numbers = np.unique(np.asarray(pub.numbers, dtype=float))
        private = env.private_state
        secret_seeds = set(_seed_values(private))
        if any(s in pub.seed_values or (abs(s) >= 10 ** 5 and s in pub.integers) for s in secret_seeds):
            seed_hits += 1
        for key, value in env.truth.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.number)) and _nontrivial(float(value)):
                truth_total += 1
                truth_hits += _contains(numbers, float(value), 1e-9)
            elif isinstance(value, str) and len(value) >= 4:
                truth_total += 1
                truth_hits += value in text
    report.files_scanned = len(seeds)
    report.exposure["seed_replay"] = {"exposed": seed_hits, "total": len(seeds)}
    report.exposure["answer_lookup"] = {"exposed": truth_hits, "total": truth_total}
    if seed_hits:
        report.findings.append(Finding("seed_replay", "high",
                                       f"episode seeds visible in {seed_hits}/{len(seeds)} public observations/states",
                                       None, seed_hits))
    if truth_hits:
        report.findings.append(Finding("answer_lookup", "high",
                                       f"{truth_hits}/{truth_total} hidden truth values visible in public "
                                       "observations/states", None, truth_hits))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python -m simulator.audit", description=__doc__.split("\n")[0])
    parser.add_argument("public", nargs="+", help="public folder(s)/file(s)")
    parser.add_argument("--private", required=True, action="append", help="run or design directory (repeatable)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = audit_export(args.public, args.private)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
