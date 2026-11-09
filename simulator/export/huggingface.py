# simulator/export/huggingface.py
"""Hugging Face dataset repository with a dataset card and Croissant metadata.

:func:`export_huggingface` turns a design directory, a run directory, or a list
of them into a folder that ``datasets.load_dataset(folder)`` reads and that can
be pushed to the Hub as-is::

    <output_dir>/
      README.md                dataset card (YAML front matter + description)
      croissant.json           Croissant 1.0 JSON-LD (MLCommons)
      data/runs.parquet        one row per run                      (config "runs")
      data/questions.parquet   one row per bundle question          (config "questions")
      images/<run>.png         bundle figures (public)
      tables/<run>.csv         data drawn in each figure (public)
      ground_truth/<run>/      ground_truth.json + exact arrays     (only with include_ground_truth)

What is public by default (the measurement side): run identity, seed, model,
fidelity level, design conditions, the configured observation model, and the
outputs whose role is ``observation`` or ``coordinate``. What is withheld unless
asked for: outputs with a ground-truth role (``parameter``, ``latent``,
``label``), ``ground_truth.json``, the run configuration (it contains the model
parameters), design columns that set a model parameter, and the question
answers *and tolerances* (a parameter question's tolerance is 5 % of the answer).

Design parameters and ground-truth parameters are stored as JSON text columns
(``design_parameters``, ``ground_truth_parameters``): their keys differ between
models and designs, are dotted config paths (``observation.operators.0.snr_db``)
and their values mix numbers, strings and booleans, so one fixed schema serves
any source and the Croissant description does not depend on the design.

Writing needs ``pyarrow`` only; ``datasets`` and ``mlcroissant`` are needed to
*use* the result.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..persistence import load_experiment_record
from ..provenance import file_sha256
from ..utils import FIDELITY_LEVELS, GROUND_TRUTH_ROLES, Role
from .tasks import collect_bundle_tasks, public_view

REPOSITORY_URL = "https://github.com/sandner-art/scientific-data-simulator"
SDS_NAMESPACE = REPOSITORY_URL + "/ns#"
CROISSANT_VERSION = "http://mlcommons.org/croissant/1.0"

from ..bundle import PRIVATE_QUESTION_KEYS  # answer and the answer-derived numeric tolerance

PUBLIC_ROLES = (Role.OBSERVATION, Role.COORDINATE)

#: Files written by the exporter; ``overwrite=True`` replaces only these.
MANAGED_PATHS = ("README.md", "croissant.json", "data", "images", "tables", "ground_truth")

_LICENSE_URLS = {
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-nc-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "mit": "https://opensource.org/licenses/MIT",
    "odc-by": "https://opendatacommons.org/licenses/by/1-0/",
}

_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

#: Croissant 1.0 JSON-LD context (as written by mlcroissant), plus the ``sds``
#: namespace for generator provenance that the Croissant/RAI vocabularies lack.
CROISSANT_CONTEXT: Dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "sds": SDS_NAMESPACE,
    "sds:generator": {"@id": "sds:generator", "@type": "@json"},
}

Source = Union[str, "os.PathLike[str]", Sequence[Union[str, "os.PathLike[str]"]]]


# ---------------------------------------------------------------------------- helpers
def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    return str(obj)


def _dumps(obj: Any) -> str:
    """Canonical JSON text (sorted keys) for JSON-valued columns."""
    return json.dumps(obj, sort_keys=True, default=_json_default)


def _scalar(value: Any) -> Any:
    """CSV cell -> plain Python value (NaN -> None)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("export_huggingface writes Parquet and needs pyarrow: "
                          "pip install 'scientific-data-simulator[parquet]'") from None
    return pa, pq


def _run_dirs(path: str) -> List[str]:
    if os.path.exists(os.path.join(path, "experiment_record.json")):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"'{path}' is neither a run directory nor a design directory")
    found = [os.path.join(path, n) for n in sorted(os.listdir(path))
             if os.path.exists(os.path.join(path, n, "experiment_record.json"))]
    if not found:
        raise FileNotFoundError(f"No run directories (with experiment_record.json) under '{path}'")
    return found


def _design_info(design_dir: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[str]]:
    """``({experiment_id: design row}, manifest, design parameter names)`` of a design directory."""
    results_path = os.path.join(design_dir, "design_results.csv")
    if not os.path.exists(results_path):
        return {}, {}, []
    manifest: Dict[str, Any] = {}
    manifest_path = os.path.join(design_dir, "design_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    table = pd.read_csv(results_path)
    columns = manifest.get("design_columns") or list(pd.read_csv(os.path.join(design_dir, "design.csv"), nrows=0).columns)
    params = [c for c in columns if c not in ("design_point", "replicate", "seed")]
    rows = {}
    for row in table.to_dict("records"):
        eid = row.get("experiment_id")
        if isinstance(eid, str) and eid:
            rows[eid] = {k: _scalar(v) for k, v in row.items()}
    return rows, manifest, params


def _value_kind(value: Any) -> Tuple[str, Any]:
    """Normalise one output value to (kind, python value) for a Parquet column."""
    if value is None:
        return "null", None
    if isinstance(value, pd.DataFrame):
        return "json", _dumps(value.to_dict(orient="list"))
    if isinstance(value, (list, tuple)):
        try:
            value = np.asarray(value)
        except ValueError:
            return "json", _dumps(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            return f"list{value.ndim}:bool", value.tolist()
        if value.dtype.kind in "iu":
            return f"list{value.ndim}:int", value.astype(np.int64).tolist()
        if value.dtype.kind == "f":
            return f"list{value.ndim}:float", value.astype(np.float64).tolist()
        if value.dtype.kind in "US":
            return f"list{value.ndim}:str", value.astype(str).tolist()
        return "json", _dumps(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return "bool", bool(value)
    if isinstance(value, (int, np.integer)):
        return "int", int(value)
    if isinstance(value, (float, np.floating)):
        return "float", float(value)
    if isinstance(value, str):
        return "str", value
    return "json", _dumps(value)


def _normalise_column(values: List[Any]) -> List[Any]:
    """One consistent type per column: ints and floats merge to float; anything
    else that disagrees between runs is stored as JSON text."""
    kinds = [_value_kind(v) for v in values]
    present = {k for k, _ in kinds if k != "null"}
    if len(present) <= 1:
        return [v for _, v in kinds]
    if present == {"int", "float"}:
        return [None if v is None else float(v) for _, v in kinds]
    if all(k.startswith("list") for k in present):
        ndims = {k.split(":")[0] for k in present}
        dtypes = {k.split(":")[1] for k in present}
        if len(ndims) == 1 and dtypes == {"int", "float"}:
            return [None if v is None else np.asarray(v, dtype=np.float64).tolist() for _, v in kinds]
    return [None if k == "null" else (v if k == "json" else _dumps(v)) for k, v in kinds]


def _size_category(n: int) -> str:
    for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
                         (1_000_000, "100K<n<1M"), (10_000_000, "1M<n<10M")):
        if n < limit:
            return label
    return "10M<n<100M"


# ---------------------------------------------------------------------------- collection
class _Run:
    __slots__ = ("run_dir", "experiment_id", "record", "raw", "ground_truth", "design_row",
                 "design_params", "design_manifest", "sort_key")


def _load_runs(source: Source) -> List[_Run]:
    sources = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    if not sources:
        raise ValueError("source is empty")
    runs: List[_Run] = []
    seen: Dict[str, str] = {}
    for src_index, src in enumerate(sources):
        src = os.path.abspath(os.fspath(src))
        group: List[_Run] = []
        design_cache: Dict[str, Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[str]]] = {}
        for run_dir in _run_dirs(src):
            parent = os.path.dirname(run_dir)
            if parent not in design_cache:
                design_cache[parent] = _design_info(parent)
            design_rows, manifest, params = design_cache[parent]
            with open(os.path.join(run_dir, "experiment_record.json"), encoding="utf-8") as f:
                raw = json.load(f)
            eid = raw["experiment_id"]
            if eid in seen:
                raise ValueError(f"Run {eid} appears twice ({seen[eid]} and {run_dir})")
            seen[eid] = run_dir
            run = _Run()
            run.run_dir = run_dir
            run.experiment_id = eid
            run.raw = raw
            # verify=True: every data file is checked against its recorded SHA-256
            run.record = load_experiment_record(parent, eid, verify=True)
            gt_path = os.path.join(run_dir, "ground_truth.json")
            if os.path.exists(gt_path):
                with open(gt_path, encoding="utf-8") as f:
                    run.ground_truth = json.load(f)
            else:
                run.ground_truth = raw.get("ground_truth") or {}
            run.design_row = design_rows.get(eid)
            run.design_params = params if run.design_row else []
            run.design_manifest = manifest if run.design_row else {}
            row = run.design_row or {}
            dp, rep = row.get("design_point"), row.get("replicate")
            run.sort_key = (src_index, 0 if dp is not None else 1, dp if dp is not None else 0,
                            rep if rep is not None else 0, os.path.basename(run_dir))
            group.append(run)
        runs.extend(group)
    runs.sort(key=lambda r: r.sort_key)
    return runs


def _model(run: _Run) -> Optional[str]:
    return (run.ground_truth.get("experiment_type") or (run.raw.get("config") or {}).get("experiment_type")
            or ".".join(p for p in (run.raw.get("experiment_logic_module"),
                                    run.raw.get("experiment_logic_class_name")) if p) or None)


def _fidelity(run: _Run) -> Optional[str]:
    return ((run.raw.get("provenance") or {}).get("fidelity") or {}).get("level")


def _split_design(run: _Run) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Design columns -> (public conditions, columns that set a model parameter)."""
    params = set((run.ground_truth.get("parameters") or {}).keys())
    public, hidden = {}, {}
    for key in run.design_params:
        value = (run.design_row or {}).get(key)
        (hidden if key.split(".")[0] in params else public)[key] = value
    return public, hidden


# ---------------------------------------------------------------------------- tables
def _runs_rows(runs: List[_Run], include_ground_truth: bool, include_seeds: bool, hide_design: bool = False
               ) -> Tuple[List[str], Dict[str, List[Any]], Dict[str, str]]:
    fixed: Dict[str, List[Any]] = {k: [] for k in (
        "experiment_id", "design_point", "replicate", "seed", "model", "fidelity",
        "simulator_version", "config_sha256", "design_parameters", "observation_model")}
    descriptions = {
        "experiment_id": "Run identifier (also the name suffix of the run directory).",
        "design_point": "Row of the design table (null for runs outside a design).",
        "replicate": "Replicate index within the design point.",
        "seed": "Seed of the run; every random draw derives from it.",
        "model": "Experiment logic class (module path).",
        "fidelity": "Model fidelity level: " + "; ".join(f"{k} = {v}" for k, v in FIDELITY_LEVELS.items()),
        "simulator_version": "Scientific Data Simulator version that generated the run.",
        "config_sha256": "SHA-256 of the canonical JSON run configuration.",
        "design_parameters": "JSON object: design conditions of the run (dotted config keys); "
                             "columns that set a model parameter are withheld unless ground truth is included.",
        "observation_model": "JSON object: configured observation model (sampling and noise operators).",
    }
    if not include_seeds:
        del fixed["seed"]
    if hide_design:
        del fixed["design_point"]
    if not include_ground_truth:
        # The hash covers the hidden parameters; for grid designs it can be brute-forced.
        del fixed["config_sha256"]
    if include_ground_truth:
        fixed.update({"ground_truth_parameters": [], "ground_truth": [], "config": []})
        descriptions.update({
            "ground_truth_parameters": "JSON object: model parameters (ground truth).",
            "ground_truth": "JSON object: ground_truth.json of the run (parameters, realised noise, law shift).",
            "config": "JSON object: full run configuration.",
        })
    roles = PUBLIC_ROLES + (GROUND_TRUTH_ROLES if include_ground_truth else ())
    order = {role: i for i, role in enumerate((Role.COORDINATE, Role.OBSERVATION) + GROUND_TRUTH_ROLES)}
    outputs: Dict[str, Tuple[int, Any]] = {}
    for run in runs:
        for name, info in run.record.output_data.items():
            d = info["descriptor"]
            if d.role in roles and name not in outputs:
                outputs[name] = (order[d.role], d)
    output_names = sorted(outputs, key=lambda n: (outputs[n][0], n))
    clash = sorted(set(output_names) & set(fixed))
    if clash:
        raise ValueError(f"Output names clash with run columns: {clash}")

    columns: Dict[str, List[Any]] = dict(fixed)
    for name in output_names:
        columns[name] = []
        d = outputs[name][1]
        bits = [f"Output '{name}', role {d.role.value}"]
        if d.units:
            bits.append(f"units {d.units}")
        if d.x_axis:
            bits.append(f"coordinate {d.x_axis}")
        descriptions[name] = ", ".join(bits) + "."
    for run in runs:
        row = run.design_row or {}
        public, _ = _split_design(run)
        prov = run.raw.get("provenance") or {}
        values = {
            "experiment_id": run.experiment_id,
            "design_point": None if row.get("design_point") is None else int(row["design_point"]),
            "replicate": None if row.get("replicate") is None else int(row["replicate"]),
            "seed": (run.raw.get("randomness") or {}).get("seed", run.ground_truth.get("seed")),
            "model": _model(run),
            "fidelity": _fidelity(run),
            "simulator_version": prov.get("simulator_version"),
            "config_sha256": prov.get("config_sha256"),
            "design_parameters": _dumps(public),
            "observation_model": _dumps((run.raw.get("config") or {}).get("observation")),
            "ground_truth_parameters": _dumps(run.ground_truth.get("parameters") or {}),
            "ground_truth": _dumps(run.ground_truth),
            "config": _dumps(run.raw.get("config") or {}),
        }
        for key in fixed:
            columns[key].append(values[key])
        for name in output_names:
            info = run.record.output_data.get(name)
            columns[name].append(None if info is None else info["data"])
    for name in output_names:
        columns[name] = _normalise_column(columns[name])
    return list(columns), columns, descriptions


def _write_parquet(path: str, names: List[str], columns: Dict[str, List[Any]],
                   types: Optional[Dict[str, Any]] = None, hf_features: Optional[Dict[str, Any]] = None) -> None:
    pa, pq = _require_pyarrow()
    types = types or {}
    arrays = [pa.array(columns[n], type=types.get(n)) for n in names]
    metadata = None
    if hf_features:
        # datasets reads per-column feature types from this key (others are inferred).
        metadata = {b"huggingface": json.dumps({"info": {"features": hf_features}}, sort_keys=True).encode()}
    table = pa.Table.from_arrays(arrays, names=names, metadata=metadata)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)


def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def _bundle_manifest(run: _Run) -> Optional[Dict[str, Any]]:
    path = os.path.join(run.run_dir, "bundle", "bundle.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _public_bundle_file(manifest: Dict[str, Any], name: Optional[str]) -> bool:
    entry = (manifest.get("files") or {}).get(name or "")
    return bool(entry) and entry.get("visibility") == "public"


def _copy_bundle_files(runs: List[_Run], output_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Copy public figures/tables; return {experiment_id: relative path} for each."""
    images, tables = {}, {}
    for run in runs:
        manifest = _bundle_manifest(run)
        if not manifest:
            continue
        figure = (manifest.get("figure") or {}).get("output")
        if figure and _public_bundle_file(manifest, figure):
            rel = f"images/{run.experiment_id}{os.path.splitext(figure)[1].lower()}"
            _copy(os.path.join(run.run_dir, "bundle", figure), os.path.join(output_dir, rel))
            images[run.experiment_id] = rel
        if _public_bundle_file(manifest, "table.csv"):
            rel = f"tables/{run.experiment_id}.csv"
            _copy(os.path.join(run.run_dir, "bundle", "table.csv"), os.path.join(output_dir, rel))
            tables[run.experiment_id] = rel
    return images, tables


def _questions(runs: List[_Run], output_dir: str, images: Dict[str, str], tables: Dict[str, str],
               include_answers: bool, include_seeds: bool, embed_images: bool
               ) -> Optional[Tuple[List[str], Dict[str, List[Any]], Dict[str, Any], Dict[str, Any], Dict[str, str]]]:
    tasks = collect_bundle_tasks([r.run_dir for r in runs])
    if not tasks:
        return None
    pa, _ = _require_pyarrow()
    names = ["id", "experiment_id", "question", "prompt", "answer_type", "choices", "tolerance_rule",
             "answerable_from", "template", "model", "fidelity", "seed", "image_path", "table_path"]
    descriptions = {
        "id": "Question identifier: <experiment_id>/<question id>.",
        "experiment_id": "Run the question is about (key into the runs config).",
        "question": "The question text.",
        "prompt": "Self-contained prompt: caption, figure hint, question and answer format.",
        "answer_type": "number, integer or choice.",
        "choices": "Allowed answers for choice questions.",
        "tolerance_rule": "How close a numeric answer must be (e.g. 3 % of the series' range); the numeric "
                          "tolerance itself is private because it is derived from the answer or the data.",
        "answerable_from": "figure (readable from the figure), inference (needs estimation) or data.",
        "template": "Question template (argmax, max, min, final, value_at, peak_count, compare_max, parameter).",
        "model": "Experiment logic class of the run.",
        "fidelity": "Fidelity level of the run's model.",
        "seed": "Seed of the run.",
        "image_path": "Figure, relative to the dataset root.",
        "table_path": "CSV with exactly the data drawn in the figure, relative to the dataset root.",
        "image": "The figure (embedded image bytes).",
        "answer": "JSON-encoded answer (number, integer or choice string).",
        "tolerance": "Absolute tolerance for numeric answers (null: exact match).",
    }
    if not include_seeds:
        names.remove("seed")
    raster = bool(images) and all(os.path.splitext(p)[1] in _IMAGE_TYPES for p in images.values())
    use_image = embed_images and raster
    if use_image:
        names.insert(names.index("image_path"), "image")
    if include_answers:
        names += ["answer", "tolerance"]
    columns: Dict[str, List[Any]] = {n: [] for n in names}
    image_cache: Dict[str, bytes] = {}
    for task in tasks:
        view = task if include_answers else {k: v for k, v in public_view(task).items()
                                             if k not in PRIVATE_QUESTION_KEYS}
        eid = view["experiment_id"]
        row = dict(view)
        row["image_path"] = images.get(eid) if task.get("image") else None
        row["table_path"] = tables.get(eid) if task.get("table") else None
        if use_image:
            rel = row["image_path"]
            if rel is None:
                row["image"] = None
            else:
                if rel not in image_cache:
                    with open(os.path.join(output_dir, rel), "rb") as f:
                        image_cache[rel] = f.read()
                row["image"] = {"bytes": image_cache[rel], "path": os.path.basename(rel)}
        if include_answers:
            row["answer"] = json.dumps(task["answer"], default=_json_default)
            row["tolerance"] = None if task.get("tolerance") is None else float(task["tolerance"])
        for n in names:
            columns[n].append(row.get(n))
    types: Dict[str, Any] = {"choices": pa.list_(pa.string()), "seed": pa.int64(), "tolerance_rule": pa.string(),
                             "image_path": pa.string(), "table_path": pa.string()}
    features: Dict[str, Any] = {}
    if use_image:
        types["image"] = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        features["image"] = {"_type": "Image"}
    if include_answers:
        types["tolerance"] = pa.float64()
    return names, columns, types, features, descriptions


# ---------------------------------------------------------------------------- Croissant
def _croissant_type(pa_type: Any) -> Tuple[Optional[str], bool]:
    """Arrow type -> (Croissant dataType, repeated); (None, _) if not describable in Croissant 1.0."""
    import pyarrow as pa
    repeated = False
    if pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
        pa_type, repeated = pa_type.value_type, True
        if pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
            return None, True  # nested lists: no Croissant 1.0 equivalent
    if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
        return "sc:Text", repeated
    if pa.types.is_boolean(pa_type):
        return "sc:Boolean", repeated
    if pa.types.is_integer(pa_type):
        return "sc:Integer", repeated
    if pa.types.is_floating(pa_type):
        return "sc:Float", repeated
    if pa.types.is_null(pa_type):
        return "sc:Text", repeated
    return None, repeated


def _cite_as() -> str:
    return ("@software{scientific_data_simulator,\n"
            "  author = {Sandner, Daniel},\n"
            "  title = {Scientific Data Simulator: A Modular Concept for Reproducible Experiment Design "
            "and Synthetic Data Generation},\n"
            f"  url = {{{REPOSITORY_URL}}}\n"
            "}")


def write_croissant(dataset_dir: str, name: str, description: str, license: str = "cc-by-4.0",
                    version: str = "1.0.0", url: Optional[str] = None,
                    date_published: Optional[str] = None, keywords: Sequence[str] = (),
                    field_descriptions: Optional[Mapping[str, Mapping[str, str]]] = None,
                    record_set_keys: Optional[Mapping[str, str]] = None,
                    rai: Optional[Mapping[str, Any]] = None,
                    generator: Optional[Mapping[str, Any]] = None,
                    cite_as: Optional[str] = None) -> str:
    """Write ``<dataset_dir>/croissant.json`` describing the Parquet files in ``data/``.

    Every ``data/<config>.parquet`` becomes a ``cr:FileObject`` (with SHA-256)
    and a record set ``<config>`` whose fields are read from the Parquet schema,
    so the description cannot drift from the data. ``images/`` and ``tables/``
    become ``cr:FileSet`` entries (images also get an ``images`` record set).
    Columns Croissant 1.0 cannot type (nested lists, embedded image structs) are
    left out of the record sets; they remain in the Parquet files.

    Args:
        rai: Croissant RAI properties, e.g. ``{"rai:dataCollection": "..."}``.
        generator: Generator provenance, stored as JSON under ``sds:generator``.
    Returns:
        Path of the written file.
    """
    _, pq = _require_pyarrow()
    field_descriptions = field_descriptions or {}
    record_set_keys = record_set_keys or {}
    data_dir = os.path.join(dataset_dir, "data")
    configs = sorted(f[:-len(".parquet")] for f in os.listdir(data_dir) if f.endswith(".parquet")) \
        if os.path.isdir(data_dir) else []
    if not configs:
        raise FileNotFoundError(f"No Parquet files in {data_dir}")

    distribution: List[Dict[str, Any]] = []
    record_sets: List[Dict[str, Any]] = []
    for config in configs:
        rel = f"data/{config}.parquet"
        file_id = f"{config}-parquet"
        distribution.append({
            "@type": "cr:FileObject", "@id": file_id, "name": file_id,
            "description": f"Parquet file of the '{config}' configuration.",
            "contentUrl": rel, "encodingFormat": "application/x-parquet",
            "sha256": file_sha256(os.path.join(dataset_dir, rel)),
        })
        schema = pq.read_schema(os.path.join(dataset_dir, rel))
        fields = []
        omitted = []
        for pa_field in schema:
            data_type, repeated = _croissant_type(pa_field.type)
            if data_type is None:
                omitted.append(pa_field.name)
                continue
            field: Dict[str, Any] = {"@type": "cr:Field", "@id": f"{config}/{pa_field.name}",
                                     "name": pa_field.name}
            text = (field_descriptions.get(config) or {}).get(pa_field.name)
            if text:
                field["description"] = text
            field["dataType"] = data_type
            if repeated:
                field["repeated"] = True
            field["source"] = {"fileObject": {"@id": file_id}, "extract": {"column": pa_field.name}}
            fields.append(field)
        record_set: Dict[str, Any] = {"@type": "cr:RecordSet", "@id": config, "name": config,
                                      "description": f"Records of {rel}."
                                      + (f" Not described here (no Croissant 1.0 type): {', '.join(omitted)}."
                                         if omitted else "")}
        if config in record_set_keys:
            record_set["key"] = {"@id": f"{config}/{record_set_keys[config]}"}
        record_set["field"] = fields
        record_sets.append(record_set)

    for folder, fmt_map in (("images", _IMAGE_TYPES), ("tables", {".csv": "text/csv"})):
        path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(path):
            continue
        exts = sorted({os.path.splitext(f)[1].lower() for f in os.listdir(path)} & set(fmt_map))
        for ext in exts:
            # FileSet and RecordSet ids must differ (JSON-LD node ids are global).
            rs_id = folder if len(exts) == 1 else f"{folder}-{ext[1:]}"
            set_id = f"{rs_id}-files"
            distribution.append({"@type": "cr:FileSet", "@id": set_id, "name": set_id,
                                 "description": f"Bundle {'figures' if folder == 'images' else 'figure data tables'}"
                                                f" ({ext[1:]}), one per run, named <experiment_id>{ext}.",
                                 "encodingFormat": fmt_map[ext], "includes": f"{folder}/*{ext}"})
            if folder == "images":
                record_sets.append({
                    "@type": "cr:RecordSet", "@id": rs_id, "name": rs_id,
                    "description": "Figures as images, keyed by file name.",
                    "field": [
                        {"@type": "cr:Field", "@id": f"{rs_id}/filename", "name": "filename",
                         "description": "File name: <experiment_id>" + ext, "dataType": "sc:Text",
                         "source": {"fileSet": {"@id": set_id}, "extract": {"fileProperty": "filename"}}},
                        {"@type": "cr:Field", "@id": f"{rs_id}/image", "name": "image",
                         "description": "The figure.", "dataType": "sc:ImageObject",
                         "source": {"fileSet": {"@id": set_id}, "extract": {"fileProperty": "content"}}},
                    ]})

    doc: Dict[str, Any] = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": CROISSANT_VERSION,
        "name": name,
        "description": description,
        "license": _LICENSE_URLS.get(license.lower(), license),
        "version": version,
    }
    if url:
        doc["url"] = url
    if date_published:
        doc["datePublished"] = date_published
    doc["citeAs"] = cite_as or _cite_as()
    if keywords:
        doc["keywords"] = list(keywords)
    doc["isAccessibleForFree"] = True
    for key, value in (rai or {}).items():
        doc[key] = value
    if generator is not None:
        doc["sds:generator"] = dict(generator)
    doc["distribution"] = distribution
    doc["recordSet"] = record_sets
    path = os.path.join(dataset_dir, "croissant.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, default=_json_default)
        f.write("\n")
    return path


# ---------------------------------------------------------------------------- card
def _yaml_front_matter(meta: Dict[str, Any]) -> str:
    import yaml
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000) + "---\n"


def _summary(runs: List[_Run]) -> Dict[str, Any]:
    def uniq(values):
        return sorted({v for v in values if v is not None}, key=str)
    prov = [r.raw.get("provenance") or {} for r in runs]
    fidelity = {}
    for p in prov:
        f = p.get("fidelity") or {}
        if f.get("level"):
            fidelity.setdefault(f["level"], f.get("meaning") or FIDELITY_LEVELS.get(f["level"], ""))
    code = {}
    for p in prov:
        c = p.get("code") or {}
        if c:
            code[(c.get("module"), c.get("class"), c.get("source_sha256"), c.get("git_commit"))] = {
                k: c.get(k) for k in ("module", "class", "source_sha256", "git_commit", "git_dirty")}
    designs = {}
    for r in runs:
        if r.design_manifest:
            designs[os.path.dirname(r.run_dir)] = r.design_manifest
    observation = uniq(_dumps((r.raw.get("config") or {}).get("observation")) for r in runs
                       if (r.raw.get("config") or {}).get("observation"))
    hidden = sorted({k for r in runs for k in _split_design(r)[1]})
    public_design = sorted({k for r in runs for k in _split_design(r)[0]})
    return {
        "n_runs": len(runs),
        "models": uniq(_model(r) for r in runs),
        "simulator_versions": uniq(p.get("simulator_version") for p in prov),
        "config_sha256": uniq(p.get("config_sha256") for p in prov),
        "fidelity": dict(sorted(fidelity.items())),
        "code": [code[k] for k in sorted(code, key=str)],
        "base_seeds": uniq(m.get("base_seed") for m in designs.values()),
        "seed_modes": uniq(m.get("seed_mode") for m in designs.values()),
        "observation_models": observation,
        "hidden_design_columns": hidden,
        "public_design_columns": public_design,
        "generated": uniq((r.raw.get("end_time") or r.raw.get("start_time") or "")[:10] or None for r in runs),
    }


def _card(name: str, pretty_name: str, description: str, license: str, summary: Dict[str, Any],
          n_questions: int, has_images: bool, include_ground_truth: bool, include_answers: bool,
          include_seeds: bool, run_columns: List[str], question_columns: List[str]) -> str:
    configs = [{"config_name": "runs", "data_files": [{"split": "train", "path": "data/runs.parquet"}],
                "default": True}]
    if n_questions:
        configs.append({"config_name": "questions",
                        "data_files": [{"split": "test", "path": "data/questions.parquet"}]})
    tags = ["synthetic", "simulation", "scientific-data", "reference-data", "scientific-data-simulator"]
    tags += [f"fidelity-{level}" for level in summary["fidelity"]]
    meta: Dict[str, Any] = {"license": license, "pretty_name": pretty_name, "tags": tags}
    if n_questions:
        meta["task_categories"] = ["visual-question-answering" if has_images else "question-answering"]
    meta["size_categories"] = [_size_category(max(summary["n_runs"], n_questions))]
    meta["configs"] = configs

    versions = ", ".join(summary["simulator_versions"]) or "unknown"
    lines = [f"# {pretty_name}", "", description, "",
             "## Dataset summary", "",
             f"Simulated (synthetic) scientific reference data: {summary['n_runs']} runs"
             + (f" and {n_questions} questions about their figures" if n_questions else "") + ", generated by "
             f"[Scientific Data Simulator]({REPOSITORY_URL}) version {versions}. "
             "No value in this dataset was measured in the real world.", "",
             "| | |", "|---|---|",
             f"| models | {', '.join(f'`{m}`' for m in summary['models']) or '-'} |",
             f"| runs | {summary['n_runs']} |"]
    if n_questions:
        lines.append(f"| questions | {n_questions} |")
    lines += [f"| fidelity levels | {', '.join(summary['fidelity']) or '-'} |",
              f"| generated | {', '.join(summary['generated']) or '-'} |", "",
              "## Configurations", "",
              "```python", "from datasets import load_dataset", "",
              f'runs = load_dataset("{name}", "runs", split="train")']
    if n_questions:
        lines.append(f'questions = load_dataset("{name}", "questions", split="test")')
    lines += ["```", "",
              f"* **runs** (`data/runs.parquet`, split `train`): one row per run. Columns: "
              + ", ".join(f"`{c}`" for c in run_columns) + ". Array outputs are list columns; "
              "`design_parameters` and `observation_model` are JSON text (`json.loads`), because their keys "
              "differ between models and designs."]
    if n_questions:
        lines.append(f"* **questions** (`data/questions.parquet`, split `test`): one row per question about a "
                     "run's figure. Columns: " + ", ".join(f"`{c}`" for c in question_columns) + "."
                     + (" `image` holds the figure (a `datasets.Image` feature); `image_path` points to the same "
                        "file under `images/`." if "image" in question_columns else ""))
    lines += ["", "## Reproducibility", "",
              "Every run records its seed; all random draws (model, observation noise, question sampling) are "
              "derived from it with `numpy.random.SeedSequence(seed).spawn(...)` (PCG64). Re-running a run's "
              "configuration with the same simulator version reproduces it exactly"
              + ("; `config_sha256` identifies the configuration." if include_ground_truth else
                 ". Seeds and configuration hashes are withheld unless the export includes them: either one, "
                 "with the simulator, would reveal the ground truth.")]
    if summary["base_seeds"]:
        lines.append("")
        lines.append(f"Design base seed(s): {', '.join(str(s) for s in summary['base_seeds'])} "
                     f"(seed mode: {', '.join(summary['seed_modes']) or '-'}); run seeds derive from the base seed."
                     if include_seeds else "Design base seeds are withheld together with the run seeds.")
    if not include_seeds:
        lines += ["", "Run seeds are **withheld** in this export: with the simulator, a seed regenerates the "
                  "observation noise and therefore the noise-free values."]
    if summary["fidelity"]:
        lines += ["", "## Fidelity levels", ""]
        lines += [f"* **{level}**: {meaning}" for level, meaning in summary["fidelity"].items()]
    lines += ["", "## Observation model", "",
              "Observations are produced from the exact model output by a configured measurement model "
              "(sampling, then noise operators such as Gaussian/Poisson noise, clipping, quantisation, missing "
              "values; missing values are NaN)."]
    if summary["public_design_columns"]:
        lines.append("Design conditions varied between runs: "
                     + ", ".join(f"`{c}`" for c in summary["public_design_columns"]) + ".")
    for obs in summary["observation_models"][:5]:
        lines += ["", "```json", obs, "```"]
    if len(summary["observation_models"]) > 5:
        lines.append(f"\n({len(summary['observation_models']) - 5} more variants in the `observation_model` column.)")
    lines += ["", "## Ground truth", ""]
    if include_ground_truth:
        lines.append("This export **includes** the ground truth: outputs with role parameter/latent/label as "
                     "columns of `runs`, `ground_truth_parameters`, `ground_truth` and `config` (JSON), and the "
                     "original files under `ground_truth/<experiment_id>/`.")
    else:
        lines.append("The ground truth is **withheld**: exact (latent) trajectories, labels and model parameters, "
                     "`ground_truth.json`, the run configurations (they contain the parameters), private bundle "
                     "files (answers, mechanism graph, description) are not part of this export.")
        if summary["hidden_design_columns"]:
            lines.append("Design columns that set a model parameter are withheld as well: "
                         + ", ".join(f"`{c}`" for c in summary["hidden_design_columns"]) + ".")
    if n_questions:
        lines.append("")
        lines.append("Question answers and grading tolerances are **included** (`answer` as JSON text, "
                     "`tolerance`)." if include_answers else
                     "Question answers and grading tolerances are **withheld** (a tolerance can reveal its "
                     "answer). Score responses with `simulator.export.tasks.score` against the private `qa.jsonl` "
                     "of the source design.")
        lines.append("")
        lines.append("Figures are public bundle files: depending on the bundle settings (`figure.show_latent`) "
                     "they may draw the exact model curves, which is what questions marked "
                     "`answerable_from: figure` rely on.")
    lines += ["", "## Citation", "", "```bibtex", _cite_as(), "```", "",
              f"Generator: {REPOSITORY_URL}. Machine-readable metadata: `croissant.json` (Croissant 1.0).", ""]
    return _yaml_front_matter(meta) + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------- main entry
def export_huggingface(source: Source, output_dir: str, name: str, license: str = "cc-by-4.0",
                       description: Optional[str] = None, include_ground_truth: bool = False,
                       include_answers: bool = False, include_seeds: bool = False,
                       pretty_name: Optional[str] = None, version: str = "1.0.0",
                       url: Optional[str] = None, date_published: Optional[str] = None,
                       embed_images: bool = True, overwrite: bool = False) -> str:
    """Export runs as a Hugging Face dataset repository with Croissant metadata.

    Args:
        source: A design directory, a run directory, or a list of either.
        output_dir: Dataset folder to write (``datasets.load_dataset(output_dir)`` loads it).
        name: Dataset name (Hub repository id, e.g. ``"org/lv-reference"``).
        license: Hugging Face license identifier.
        description: One-paragraph description for the card and Croissant.
        include_ground_truth: Add latent/parameter/label outputs, ground-truth JSON
            columns and ``ground_truth/`` files.
        include_answers: Add ``answer`` and ``tolerance`` to the questions.
        include_seeds: Publish run seeds (and design base seeds). Off by default: a
            seed plus the simulator regenerates a run, answers included. Turn it on
            for open, fully reproducible datasets whose answers are not secret.
        version: Dataset version (MAJOR.MINOR.PATCH).
        url: Dataset URL for Croissant (e.g. its Hub page).
        date_published: ISO date; default: the date the last run finished.
        embed_images: Store figures in ``questions.parquet`` as a ``datasets.Image``
            feature (bytes) in addition to ``image_path``.
        overwrite: Replace an existing export in ``output_dir`` (only the files this
            exporter writes are removed).

    Returns:
        ``output_dir``.
    """
    _require_pyarrow()
    runs = _load_runs(source)
    # When design columns are withheld, the design-point index and the run order (execution
    # order follows the design table) would map runs back to the withheld conditions of a
    # public or regenerable design: drop the index and order runs by their random ids.
    hide_design = not include_ground_truth and any(_split_design(run)[1] for run in runs)
    if hide_design:
        runs = sorted(runs, key=lambda run: run.experiment_id)
    output_dir = os.path.abspath(output_dir)
    if os.path.isdir(output_dir):
        existing = [p for p in MANAGED_PATHS if os.path.exists(os.path.join(output_dir, p))]
        if existing and not overwrite:
            raise FileExistsError(f"{output_dir} already contains an export ({', '.join(existing)}); "
                                  "pass overwrite=True to replace it")
        for p in existing:
            full = os.path.join(output_dir, p)
            shutil.rmtree(full) if os.path.isdir(full) else os.remove(full)
    os.makedirs(output_dir, exist_ok=True)
    summary = _summary(runs)
    pretty_name = pretty_name or name.split("/")[-1].replace("-", " ").replace("_", " ")
    description = description or (
        f"Simulated scientific reference data from {', '.join(summary['models']) or 'a simulation model'}: "
        "noisy, sampled observations of known mechanisms, generated reproducibly by Scientific Data Simulator.")

    run_names, run_columns, run_desc = _runs_rows(runs, include_ground_truth, include_seeds, hide_design)
    pa, _ = _require_pyarrow()
    run_types = {"design_point": pa.int64(), "replicate": pa.int64(), "seed": pa.int64()}
    for key in ("model", "fidelity", "simulator_version", "config_sha256", "observation_model"):
        run_types[key] = pa.string()
    _write_parquet(os.path.join(output_dir, "data", "runs.parquet"), run_names, run_columns, run_types)

    images, tables = _copy_bundle_files(runs, output_dir)
    q = _questions(runs, output_dir, images, tables, include_answers, include_seeds, embed_images)
    q_names: List[str] = []
    n_questions = 0
    descriptions = {"runs": run_desc}
    if q:
        q_names, q_columns, q_types, q_features, q_desc = q
        n_questions = len(q_columns["id"])
        _write_parquet(os.path.join(output_dir, "data", "questions.parquet"), q_names, q_columns,
                       q_types, q_features)
        descriptions["questions"] = q_desc

    if include_ground_truth:
        for run in runs:
            dst = os.path.join(output_dir, "ground_truth", run.experiment_id)
            gt_path = os.path.join(run.run_dir, "ground_truth.json")
            if os.path.exists(gt_path):
                _copy(gt_path, os.path.join(dst, "ground_truth.json"))
            for entry in (run.raw.get("output_data") or {}).values():
                storage = entry.get("storage") or {}
                role = (entry.get("descriptor") or {}).get("role")
                if storage.get("path") and role in {r.value for r in GROUND_TRUTH_ROLES}:
                    _copy(os.path.join(run.run_dir, storage["path"]),
                          os.path.join(dst, *storage["path"].split("/")))

    date_published = date_published or (summary["generated"][-1] if summary["generated"] else None)
    fidelity_text = "; ".join(f"{k}: {v}" for k, v in summary["fidelity"].items())
    rai = {
        "rai:dataCollection": (
            "Synthetic data: every record was generated by simulation with Scientific Data Simulator "
            f"({REPOSITORY_URL}, version {', '.join(summary['simulator_versions']) or 'unknown'}). Exact model "
            "outputs are computed from known mechanisms and parameters; observations are derived from them by a "
            "configured, seeded measurement model. No real-world measurement or personal data is involved."),
        "rai:dataCollectionType": ["Simulation"],
        "rai:dataCollectionRawData": "None: all values are simulated; the generating configuration and seed of "
                                     "each run are recorded.",
        "rai:dataCollectionMissingData": "Missing observations (from the measurement model's missing-value "
                                         "operator) are NaN in observation arrays.",
        "rai:dataLimitations": "Simulated data reflect the model, not nature. Fidelity levels: "
                               + (fidelity_text or "not recorded") + ".",
        "rai:dataUseCases": "Validation of analysis, inference and AI methods against known ground truth; "
                            "figure question answering.",
        "rai:personalSensitiveInformation": "None (synthetic data).",
    }
    generator = {
        "synthetic": True,
        "simulator": "scientific-data-simulator",
        "repository": REPOSITORY_URL,
        "simulator_versions": summary["simulator_versions"],
        "models": summary["models"],
        "fidelity_levels": list(summary["fidelity"]),
        "code": summary["code"],
        "config_sha256": summary["config_sha256"] if include_ground_truth else None,
        "base_seeds": summary["base_seeds"] if include_seeds else None,
        "seed_derivation": "numpy.random.SeedSequence(seed).spawn(n_streams), PCG64",
        "n_runs": summary["n_runs"],
        "n_questions": n_questions,
        "ground_truth_included": include_ground_truth,
        "answers_included": include_answers,
        "seeds_included": include_seeds,
        "withheld_design_columns": [] if include_ground_truth else summary["hidden_design_columns"],
    }
    write_croissant(output_dir, name=name, description=description, license=license, version=version,
                    url=url, date_published=date_published,
                    keywords=["synthetic", "simulation", "scientific data", "reference data"],
                    field_descriptions=descriptions,
                    record_set_keys={"runs": "experiment_id", "questions": "id"},
                    rai=rai, generator=generator)
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(_card(name, pretty_name, description, license, summary, n_questions, bool(images),
                      include_ground_truth, include_answers, include_seeds, run_names, q_names))
    return output_dir
