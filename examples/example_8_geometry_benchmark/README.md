# Example 8: a triangle-geometry benchmark

This example generates random triangles with exact constructions: circumcircle, incircle, centroid, orthocentre and the Euler line. It draws each one as a diagram, asks questions with exact answers, and exports everything as a Hugging Face dataset with Croissant metadata.

```bash
python -m examples.example_8_geometry_benchmark.run_experiment --n 20
python -m examples.example_8_geometry_benchmark.run_experiment --n 20 --bare        # diagrams without axes
python -m examples.example_8_geometry_benchmark.run_experiment --n 20 --no-export
```

## What it shows

- **A diagram model.** `TriangleExperiment` returns exact coordinates and measurements, and describes its drawing through `figure_elements()`. The bundle writes `elements.csv` and a `plot_figure.py` that draws the diagram from it, then executes that script to make `figure.png`.
- **Model-specific questions.** `questions()` asks for the area, perimeter, an angle, the circumradius and inradius, the triangle type, the longest side, and the distance between the circumcentre and the centroid. Answers are exact. Public files show only a tolerance *rule*, such as "3 % of the true value", because the numeric tolerance would reveal the answer.
- **Noisy measurements.** The side lengths are also "measured" with a ruler, with Gaussian noise and 1 mm quantisation (`side_lengths_observed`), for inference tasks.
- **Verification.** `tests/test_geometry.py` checks every triangle against independent identities:
  - Heron's formula against the shoelace area;
  - the law of sines;
  - Euler's theorem OI² = R(R − 2r);
  - the Euler-line ratio OG : GH = 1 : 2;
  - the angle sum.
- **Hugging Face export.** `hf_dataset/` holds `README.md` (dataset card), `croissant.json`, `data/runs.parquet`, `data/questions.parquet` (with the figures embedded) and `images/`. Answers are withheld; pass `include_answers=True` to `export_huggingface` for a private copy.

## Output

```
experiments_output/example_8_geometry_benchmark/
├── design.csv, design_manifest.json, design_results.csv
├── questions.jsonl / qa.jsonl         all questions (public / with answers)
├── <run>/bundle/                      figure.png, plot_figure.py, elements.csv, caption.txt, ...
└── hf_dataset/                        README.md, croissant.json, data/*.parquet, images/
```
