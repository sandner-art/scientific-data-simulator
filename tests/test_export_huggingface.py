# tests/test_export_huggingface.py
import json
import os

import numpy as np
import pytest

from simulator.provenance import simulator_version
import yaml

from simulator.config import load_config
from simulator.doe import create_doe_table, run_design
from simulator.export.huggingface import PRIVATE_QUESTION_KEYS, export_huggingface
from simulator.provenance import file_sha256

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "examples", "example_7_reference_dataset", "config.yaml")


def _base(**overrides):
    base = load_config(CONFIG)
    base.update({"n_steps": 200, "bundle": True, "generate_plots": False})
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def design(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("design"))
    run_design(_base(), create_doe_table({"observation.operators.0.snr_db": [30, 10]}),
               output_dir=out, replicates=2, seed_mode="per_replicate")
    return out


@pytest.fixture(scope="module")
def param_design(tmp_path_factory):
    """A design that sweeps a model parameter, without bundles."""
    out = str(tmp_path_factory.mktemp("param_design"))
    run_design(_base(bundle=False), create_doe_table({"prey_growth_rate": [1.0, 1.2]}), output_dir=out)
    return out


@pytest.fixture(scope="module")
def public(design, tmp_path_factory):
    return export_huggingface(design, str(tmp_path_factory.mktemp("hf") / "public"), "org/lv-reference")


@pytest.fixture(scope="module")
def full(design, tmp_path_factory):
    return export_huggingface(design, str(tmp_path_factory.mktemp("hf") / "full"), "org/lv-reference",
                              include_ground_truth=True, include_answers=True, include_seeds=True)


def _front_matter(path):
    text = open(os.path.join(path, "README.md"), encoding="utf-8").read()
    assert text.startswith("---\n")
    head, body = text[4:].split("\n---\n", 1)
    return yaml.safe_load(head), body


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _all_files(path):
    return sorted(os.path.relpath(os.path.join(d, f), path).replace(os.sep, "/")
                  for d, _, files in os.walk(path) for f in files)


# ------------------------------------------------------------------ layout and card
def test_layout(public):
    files = _all_files(public)
    assert {"README.md", "croissant.json", "data/runs.parquet", "data/questions.parquet"} <= set(files)
    assert len([f for f in files if f.startswith("images/")]) == 4
    assert not [f for f in files if f.startswith("tables/")]  # exact tables are private by default
    assert not any(f.startswith("ground_truth/") for f in files)


def test_dataset_card(public):
    meta, body = _front_matter(public)
    assert meta["license"] == "cc-by-4.0"
    assert meta["pretty_name"]
    assert "synthetic" in meta["tags"]
    assert meta["size_categories"] == ["n<1K"]
    assert meta["task_categories"] == ["visual-question-answering"]
    configs = {c["config_name"]: c for c in meta["configs"]}
    assert configs["runs"]["data_files"] == [{"split": "train", "path": "data/runs.parquet"}]
    assert configs["questions"]["data_files"] == [{"split": "test", "path": "data/questions.parquet"}]
    for text in (simulator_version(), "reference", "seed", "withheld", "github.com/sandner-art/scientific-data-simulator"):
        assert text in body


# ------------------------------------------------------------------ runs
def test_runs_public_by_default(design, public):
    table = pq.read_table(os.path.join(public, "data", "runs.parquet"))
    names = set(table.column_names)
    assert {"time", "time_observed", "prey_population_observed", "predator_population_observed"} <= names
    # ground truth (latent arrays, parameters, configuration) is not exported
    assert not names & {"prey_population", "predator_population", "config", "ground_truth",
                        "ground_truth_parameters", "observation_index"}
    rows = table.to_pylist()
    results = _jsonl_free_results(design)
    assert [r["experiment_id"] for r in rows] == results["experiment_id"]
    # seeds and configuration hashes would let anyone regenerate the hidden truth
    assert "seed" not in names and "config_sha256" not in names
    assert [json.loads(r["design_parameters"]) for r in rows] == \
        [{"observation.operators.0.snr_db": v} for v in results["observation.operators.0.snr_db"]]
    assert {r["fidelity"] for r in rows} == {"reference"}
    assert all(r["model"] == "experiments.predator_prey.logic.PredatorPreyExperiment" for r in rows)
    assert len(rows[0]["time_observed"]) == len(rows[0]["prey_population_observed"])


def _jsonl_free_results(design):
    import pandas as pd
    table = pd.read_csv(os.path.join(design, "design_results.csv"))
    table = table.sort_values(["design_point", "replicate"])
    return {c: table[c].tolist() for c in table.columns}


def test_design_columns_that_set_parameters_are_withheld(param_design, tmp_path):
    out = export_huggingface(param_design, str(tmp_path / "hf"), "org/params")
    rows = pq.read_table(os.path.join(out, "data", "runs.parquet")).to_pylist()
    assert [json.loads(r["design_parameters"]) for r in rows] == [{}, {}]
    assert not os.path.exists(os.path.join(out, "data", "questions.parquet"))  # no bundles
    meta, body = _front_matter(out)
    assert [c["config_name"] for c in meta["configs"]] == ["runs"]
    assert "task_categories" not in meta
    assert "`prey_growth_rate`" in body
    croissant = json.load(open(os.path.join(out, "croissant.json")))
    assert croissant["sds:generator"]["withheld_design_columns"] == ["prey_growth_rate"]

    full = export_huggingface(param_design, str(tmp_path / "full"), "org/params", include_ground_truth=True)
    rows = pq.read_table(os.path.join(full, "data", "runs.parquet")).to_pylist()
    assert [json.loads(r["ground_truth_parameters"])["prey_growth_rate"] for r in rows] == [1.0, 1.2]


# ------------------------------------------------------------------ questions
def test_questions_without_answers(design, public):
    table = pq.read_table(os.path.join(public, "data", "questions.parquet"))
    assert not set(table.column_names) & set(PRIVATE_QUESTION_KEYS)
    rows = table.to_pylist()
    qa = _jsonl(os.path.join(design, "qa.jsonl"))
    assert len(rows) == len(qa)
    assert {r["id"] for r in rows} == {f"{q['experiment_id']}/{q['id']}" for q in qa}
    for row in rows:
        assert row["image_path"].startswith("images/")
        with open(os.path.join(public, row["image_path"]), "rb") as f:
            assert row["image"]["bytes"] == f.read()
        assert row["table_path"] is None
        assert "Question:" in row["prompt"]
    choice = [r for r in rows if r["answer_type"] == "choice"]
    assert choice and all(len(r["choices"]) == 2 for r in choice)
    # Hugging Face reads the image column as a datasets.Image feature
    meta = json.loads(table.schema.metadata[b"huggingface"])
    assert meta["info"]["features"] == {"image": {"_type": "Image"}}


def test_ground_truth_and_answers_on_request(design, full):
    runs = pq.read_table(os.path.join(full, "data", "runs.parquet")).to_pylist()
    assert {"prey_population", "predator_population"} <= set(runs[0])
    params = json.loads(runs[0]["ground_truth_parameters"])
    assert params["prey_growth_rate"] == 1.1
    assert json.loads(runs[0]["config"])["seed"] == runs[0]["seed"]

    questions = pq.read_table(os.path.join(full, "data", "questions.parquet")).to_pylist()
    qa = {f"{q['experiment_id']}/{q['id']}": q for q in _jsonl(os.path.join(design, "qa.jsonl"))}
    for row in questions:
        assert json.loads(row["answer"]) == qa[row["id"]]["answer"]
        assert row["tolerance"] == qa[row["id"]]["tolerance"]

    for run in runs:
        gt_dir = os.path.join(full, "ground_truth", run["experiment_id"])
        record_dir = [d for d in os.listdir(design) if d.endswith(run["experiment_id"])][0]
        record = json.load(open(os.path.join(design, record_dir, "experiment_record.json")))
        assert os.path.exists(os.path.join(gt_dir, "ground_truth.json"))
        latent = record["output_data"]["prey_population"]["storage"]
        copied = os.path.join(gt_dir, *latent["path"].split("/"))
        assert file_sha256(copied) == latent["sha256"]
        np.testing.assert_array_equal(np.load(copied), np.asarray(run["prey_population"]))


def test_seeds_can_be_withheld(design, tmp_path):
    out = export_huggingface(design, str(tmp_path / "hf"), "org/sealed", include_seeds=False)
    assert "seed" not in pq.read_schema(os.path.join(out, "data", "runs.parquet")).names
    assert "seed" not in pq.read_schema(os.path.join(out, "data", "questions.parquet")).names
    croissant = json.load(open(os.path.join(out, "croissant.json")))
    assert croissant["sds:generator"]["base_seeds"] is None
    assert "20261108" not in open(os.path.join(out, "README.md")).read()


# ------------------------------------------------------------------ Croissant
def test_croissant_metadata(public):
    doc = json.load(open(os.path.join(public, "croissant.json"), encoding="utf-8"))
    assert doc["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert doc["@type"] == "sc:Dataset"
    assert doc["license"] == "https://creativecommons.org/licenses/by/4.0/"
    assert doc["name"] == "org/lv-reference" and doc["version"] == "1.0.0"
    assert "rai:dataCollection" in doc and "Synthetic" in doc["rai:dataCollection"]
    gen = doc["sds:generator"]
    assert gen["synthetic"] is True and gen["simulator_versions"] == [simulator_version()]
    assert gen["base_seeds"] is None and gen["config_sha256"] is None  # private by default

    files = {d["@id"]: d for d in doc["distribution"]}
    for config in ("runs", "questions"):
        entry = files[f"{config}-parquet"]
        assert entry["@type"] == "cr:FileObject"
        assert file_sha256(os.path.join(public, entry["contentUrl"])) == entry["sha256"]
    assert files["images-files"]["@type"] == "cr:FileSet"
    assert files["images-files"]["includes"] == "images/*.png"

    record_sets = {r["@id"]: r for r in doc["recordSet"]}
    assert set(record_sets) == {"runs", "questions", "images"}
    for config in ("runs", "questions"):
        columns = pq.read_schema(os.path.join(public, "data", f"{config}.parquet")).names
        fields = record_sets[config]["field"]
        described = [f["source"]["extract"]["column"] for f in fields]
        assert described == [c for c in columns if c != "image"]
        assert all(f["source"]["fileObject"]["@id"] == f"{config}-parquet" for f in fields)
    by_name = {f["name"]: f for f in record_sets["runs"]["field"]}
    assert "seed" not in by_name and by_name["design_point"]["dataType"] == "sc:Integer"
    assert by_name["prey_population_observed"]["dataType"] == "sc:Float"
    assert by_name["prey_population_observed"]["repeated"] is True


def test_export_is_deterministic(design, public, tmp_path):
    again = export_huggingface(design, str(tmp_path / "again"), "org/lv-reference")
    files = _all_files(public)
    assert files == _all_files(again)
    for name in files:
        assert file_sha256(os.path.join(public, name)) == file_sha256(os.path.join(again, name)), name


def test_overwrite_and_sources(design, param_design, tmp_path):
    out = str(tmp_path / "hf")
    export_huggingface(param_design, out, "org/a")
    with pytest.raises(FileExistsError):
        export_huggingface(param_design, out, "org/a")
    run_dir = os.path.join(design, sorted(d for d in os.listdir(design)
                                          if os.path.isdir(os.path.join(design, d)))[0])
    export_huggingface([run_dir, param_design], out, "org/a", overwrite=True)
    rows = pq.read_table(os.path.join(out, "data", "runs.parquet")).to_pylist()
    assert len(rows) == 3
    assert os.path.exists(os.path.join(out, "data", "questions.parquet"))  # the run dir has a bundle
    with pytest.raises(ValueError, match="twice"):
        export_huggingface([design, design], str(tmp_path / "dup"), "org/a")


# ------------------------------------------------------------------ consumers (optional)
def test_mlcroissant_validates_and_reads(public):
    mlc = pytest.importorskip("mlcroissant")
    dataset = mlc.Dataset(jsonld=os.path.join(public, "croissant.json"))
    assert not dataset.metadata.ctx.issues.errors
    assert not dataset.metadata.ctx.issues.warnings
    counts = {rs.uuid: len(list(dataset.records(rs.uuid))) for rs in dataset.metadata.record_sets}
    n_questions = pq.read_metadata(os.path.join(public, "data", "questions.parquet")).num_rows
    assert counts == {"runs": 4, "questions": n_questions, "images": 4}


def test_datasets_loads_each_config(public):
    datasets = pytest.importorskip("datasets")
    runs = datasets.load_dataset(public, "runs", split="train")
    assert runs.num_rows == 4
    assert isinstance(runs[0]["prey_population_observed"], list)
    questions = datasets.load_dataset(public, "questions", split="test")
    assert isinstance(questions.features["image"], datasets.Image)
    image = questions[0]["image"]
    assert image.size == (768, 576)
    assert datasets.load_dataset(public, split="train").num_rows == 4  # "runs" is the default config


def test_public_tables_are_exported_when_marked_public(tmp_path):
    out = str(tmp_path / "design")
    run_design(_base(bundle={"public_table": True}), create_doe_table({"observation.operators.0.snr_db": [30]}),
               output_dir=out, base_seed=5)
    hf = export_huggingface(out, str(tmp_path / "hf"), "org/training")
    assert len(os.listdir(os.path.join(hf, "tables"))) == 1
    rows = pq.read_table(os.path.join(hf, "data", "questions.parquet")).to_pylist()
    assert all(os.path.exists(os.path.join(hf, r["table_path"])) for r in rows)
