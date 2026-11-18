# Example 13: a frozen law-discovery release

Builds one fixed law-discovery dataset to publish and cite. It runs the suite of example 10 under
secret seeds, exports it for the Hugging Face Hub with the answers withheld, adds reference
baselines and a canary to the dataset card, audits the public folder for leaks, and writes
checksums. The answer key stays in a private folder.

```bash
python -m examples.example_13_law_release.run_experiment --output_dir release           # about 2 minutes
python -m examples.example_13_law_release.run_experiment --output_dir release --quick   # about 20 s, a trial
```

By default: 9 families x textbook/counterfactual x 40 and 20 dB = 36 design points, 3 laws each:
108 runs and 324 questions (law family, formula, extrapolated value). The generic chart
read-outs (maximum, minimum, final value) are left out: they are not law discovery, and the
minimum was answerable without the data two times in three in a trial build.

## What it writes

| Folder | Contents | Publish? |
|---|---|---|
| `public/` | `data/runs.parquet` (measurements), `data/questions.parquet` (questions, no answers), `images/` (figures), `README.md` (dataset card with baselines), `croissant.json`, `CANARY.txt`, `SHA256SUMS` | **yes** |
| `private/` | `runs/` and `answer_key.jsonl` (the answers), `release.json` (secret seeds, canary, checksum of `SHA256SUMS`), `audit.json`, `REPORT.md`, `twin/` (a fresh twin, never released, for contamination tests), `scores/`, `llm/` | **never** |

The build stops before writing checksums if the leak audit of `public/` finds anything. With the
seeds anyone could regenerate the runs and their answers, so they exist only in `release.json`.

## Publishing

1. Check the folder: `cd release/public && sha256sum -c SHA256SUMS`.
2. Upload it as a dataset repository (needs a Hugging Face account and `huggingface_hub`):

   ```bash
   huggingface-cli login
   huggingface-cli upload <user-or-org>/sds-law-discovery release/public . --repo-type dataset
   ```

3. Get a DOI: on the dataset page, *Settings* > *Generate DOI*. Or upload the same folder to
   Zenodo, which also archives it. Put the DOI in the dataset card and in the paper.
4. Keep `private/` somewhere safe and off the Hub. Without it the release cannot be scored.

Rebuilding does not reproduce a release: the seeds are drawn afresh. To publish a new version,
build again with `--version`.

## Scoring

```bash
# predictions from anywhere: JSON lines {"id": ..., "response": ...}, free text accepted
python -m examples.example_13_law_release.run_experiment --output_dir release --score preds.jsonl --label my-model

# or ask a model directly; every call is logged (private/llm/) and can be replayed without the model
python -m examples.example_13_law_release.run_experiment --output_dir release \
    --solve mymodels.client:ask --model-name provider/model-2026-11
```

`mymodels.client:ask` is any function `prompt -> text`. Scores go to `private/scores/`, per
question type, for formulas also without the items a fitted polynomial passes
(`polynomial_reachable`), and for extrapolations also without the items that pass with the answer 0
or with a naive extrapolation (`zero_reachable`, `naive_reachable`: laws that have decayed, or that
barely change beyond the data, such as plateaus).

## Baselines in a trial build

A full build (108 runs, 324 questions; 6 MB public, 25 MB private including the twin) gave:

| solver | law family | formula | extrapolation | formula without polynomial-reachable items |
|---|---|---|---|---|
| curve fitting (all families, best by AIC) | 0.88 | 0.88 | 0.94 | 0.90 |
| answering without the data | 0.00 | 0.00 | 0.00 | |

A fitted polynomial passed 7 of the 108 formula questions (0.06). The numbers change from build to
build, since the laws are drawn afresh; the dataset card of each release records its own.
