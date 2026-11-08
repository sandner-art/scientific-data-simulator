# Example 7: a predator–prey reference dataset

This example builds a small benchmark-style dataset. The predator–prey model is integrated with RK4 (verified in `tests/test_verification.py`), so its trajectories serve as ground truth. Observations are sampled from those trajectories and made noisy, at three noise levels with several seeded replicates each.

```bash
python -m examples.example_7_reference_dataset.run_experiment
python -m examples.example_7_reference_dataset.run_experiment --law-shift   # counterfactual law
python -m examples.example_7_reference_dataset.run_experiment --replicates 10 --n_jobs 4
```

## What it shows

- **One recorded seed**, `seed: 20261108` in `config.yaml`, determines the whole dataset. Each run's own seed is listed in `design.csv`, so any single run can be repeated on its own.
- **Observation operators** are applied after the mechanism. The run samples every 25th step, adds Gaussian noise at a set signal-to-noise ratio (SNR), clips at 0 and marks 5 % of values as missing.
- **A sweep over a nested config key**: `observation.operators.0.snr_db` takes the values 30, 20 and 10 dB.
- **Common random numbers** (`seed_mode="per_replicate"`): within one replicate, the three noise levels observe the same latent trajectory.
- **Ground truth kept apart from observations**: `data/ground_truth/`, `data/observation/` and `ground_truth.json`. The answers are not mixed in with what a method under test gets to see.
- **Multimodal bundles** (`bundle:`): every run gets a figure rendered by its own `plot_figure.py`, the plotted `table.csv`, a caption, verifiable Q&A and the mechanism graph. The design directory collects all questions into `questions.jsonl` (public) and `qa.jsonl` (private, with answers). Set `bundle.figure.show_latent: false` to draw observations only; the exact-curve questions then become inference tasks.
- **Optional law shift** (`--law-shift`): predation switches to a saturating Holling type II response. The original and the shifted settings are both recorded in `ground_truth.json`.

## Output

```
experiments_output/example_7_reference_dataset/
├── design.csv               design rows with seeds (written before running)
├── design_manifest.json     base seed, seed mode, base config
├── design_results.csv       design + experiment_id + status, one row per run
├── questions.jsonl          all bundle questions, without answers (public)
├── qa.jsonl                 the same with answers and tolerances (private)
└── <timestamp>_<id>/
    ├── experiment_record.json   provenance and a checksum manifest of the data files
    ├── ground_truth.json        parameters, realised noise levels, law shift
    ├── data/ground_truth/       prey_population.npy, predator_population.npy
    ├── data/observation/        *_observed.npy (sampled and noisy)
    ├── data/coordinate/         time.npy, time_observed.npy
    ├── bundle/                  figure.png, plot_figure.py, table.csv, caption.txt, questions.json,
    │                            qa.json, graph.json, graph.dot, description.md, bundle.json
    └── time_series_*.png/html   latent curves with observed points overlaid
```

To share an observations-only version, for example as a held-out benchmark, copy each run's `data/observation/` and `data/coordinate/` plus the bundle files marked `public` in `bundle.json`. Keep `ground_truth.json`, `qa.json` and the graph private.
