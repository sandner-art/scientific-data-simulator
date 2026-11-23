# Reproducing the SciLaws-Bench audit

`tools/audit_scilaws.py` measures SciLaws-Bench (Huang et al., arXiv:2609.01552) with scidatasim's
instruments:

- **The leak audit** (`scidatasim.audit.audit_export`) on every task, with an answer key built from
  the task's hidden law (`simulator/formula.py`, read with `ast`, never run) and the published laws'
  fitted constants (`eval/reference_metrics.json`). It runs on the files a solver sees
  (`metadata.yaml`, `data/`) and, as a positive control, on the files the release itself documents as
  answer keys (`eval/`, `simulator/`). Besides that:
  - `simulator/state.joblib` is read as pickle opcodes and never unpickled;
  - train/test overlap and the share of test inputs outside the training range are counted;
  - every text file is searched for a canary string.
- **Generic non-law submissions**: a constant, polynomials and additive natural cubic splines, raw or
  log. For Type II tasks also per-group polynomials, and one global shape with a per-group offset
  and/or scale.
  - They are fitted on `train.csv` only, with the variant chosen by cross-validation that imitates
    the test split.
  - They are written in the benchmark's submission format and scored by its own harness
    (`harness/evaluate_numeric.py score`, one child process per submission).
  - Each is held to three limits on its fitted numbers: what the scorer checks; the anti-hacking
    judge's counting rule (caps + 3); and the task prompt's caps.

Nothing from the release is executed except the benchmark's own scorer, on submissions the tool
writes. The method is described in full in the tool's docstring.

## What you need

| | Version used on 23 November 2026 |
|---|---|
| scidatasim | this repository: commit 876c18e (patch 0058) for run A, 1172ae8 (patch 0060) for run B |
| SciLaws-Bench code (harness, published agent scores) | `https://github.com/yiyihum/SciLaws-Bench`, commit `8174c7b5289fd3e414892f20efca322cb1e0e5e9` |
| SciLaws-Bench data | Hugging Face dataset `RealSR/SciLaws-Bench`, `tasks/` only (1.1 GB), revision `<REVISION>` |
| Python | 3.12.10 (any 3.10-3.13 should do), with scidatasim's dependencies (numpy 2.2.6, pandas, scipy, PyYAML) |
| huggingface_hub | 0.36 (`hf` command). In an environment that also holds transformers 4.57, keep it below 1.0 |
| Machine | Windows 11 desktop; Linux works with the same commands |
| Time | 76 minutes for the 118 tasks (one task, `latent_heat_flux_kgdsr__LE`, takes 39 of them) |

## Commands

Linux or macOS:

```bash
pip install -e .                                  # in this repository; or: pip install scidatasim
hf download RealSR/SciLaws-Bench --repo-type dataset --revision <REVISION> \
    --local-dir scilaws_data --include "tasks/*"
git clone https://github.com/yiyihum/SciLaws-Bench scilaws_code
git -C scilaws_code checkout 8174c7b5289fd3e414892f20efca322cb1e0e5e9
python tools/audit_scilaws.py --tasks scilaws_data/tasks --harness scilaws_code --out scilaws_out
```

Windows (cmd):

```bat
pip install -e .
hf download RealSR/SciLaws-Bench --repo-type dataset --revision <REVISION> --local-dir C:\opendata\scilaws_data --include "tasks/*"
git clone https://github.com/yiyihum/SciLaws-Bench C:\opendata\scilaws_code
git -C C:\opendata\scilaws_code checkout 8174c7b5289fd3e414892f20efca322cb1e0e5e9
python tools\audit_scilaws.py --tasks C:\opendata\scilaws_data\tasks --harness C:\opendata\scilaws_code --out C:\opendata\scilaws_out
```

An interrupted run continues with `--resume` (finished tasks are kept in `<out>/tasks/`). `--jobs N`
sets how many scorer processes run at once (default: half the CPU cores).

**Which data revision.** If you downloaded without `--revision`, `hf download` records the commit it
fetched. It is the first line of any `.metadata` file under the download's cache, for example:

```
<local-dir>/.cache/huggingface/download/tasks/typeI/baseball_pythagorean_winpct_lahman__win_pct/metadata.yaml.metadata
```

## Windows

The scorer needed two adjustments on Windows. Both are built into the tool, so nothing has to be done
by hand.

- **Encoding.** The scorer opens `metadata.yaml` and the CSV files without naming an encoding. On
  Windows that means cp1252, and a task file holding, for example, `log₁₀` (U+2081 contains the byte
  0x81, which cp1252 does not define) stops it with `UnicodeDecodeError`. Without the fix, 20 of the
  118 tasks failed. The tool runs the scorer in UTF-8 mode (`python -I -X utf8`). `-I` ignores the
  `PYTHONUTF8` variable, so the flag is passed on the command line.
- **Fit timeout.** The scorer limits each per-group fit with `signal.alarm`, which Windows does not
  have. There the tool makes the alarm a no-op and instead times every fit itself. The report gives
  the slowest fit against the task's cap (on 23 November: no fit came near a cap).

The tool's own reads always name UTF-8. If the scorer crashes for any other reason, the report says
so ("The harness crashed on ..."). It then uses S_N computed by the tool for those submissions (the
same formula) instead of scoring them 0.

## What a correct run shows

In `<out>/scilaws_audit.md` and `.json`:

- `harness.matches_pin`: true (the scorer is at commit 8174c7b).
- "S_N computed here agrees with the harness on N of N submissions both scored", with no crashed
  submissions. On 23 November: 1,955 of 1,955.
- The positive control: the hidden law found in the grader-facing files in 118 of 118 tasks.

The other files: `tasks/<task>.json` holds each task's full record. `submissions/<task>/*.py` are the
scored submissions, which can be rescored by hand with
`python scilaws_code/harness/evaluate_numeric.py score <task folder> <submission>`.
`answer_keys/<task>.jsonl` are the keys the audit used.

## Results of 23 November 2026

S_N is SciLaws' numeric score: 0.5 means the test error equals the best published law's, and 1
means a perfect prediction. "Pass" is S_N ≥ 0.5. The agent scores are the ones published with the
code (`baseline_agent/model_task_scores.json`).

| Solver | Mean S_N | Pass (of 118) |
|---|---:|---:|
| Curve fit, only what the scorer checks | 0.441 | 57 |
| Curve fit, anti-hacking judge's count (caps + 3) | 0.390 | 43 |
| Curve fit, prompt's caps | 0.316 | 25 |
| gpt-5.5 | 0.508 | 72 |
| glm-5.2, gemini-3.5-flash-fg, claude-opus-4-8-fg, deepseek-v4-pro, qwen3.7-max | 0.438-0.461 | 58-65 |
| gpt-5-mini, gpt5.4-mini | 0.407, 0.372 | 43, 40 |
| gpt-4o-mini | 0.206 | 18 |

The fitting is seeded per task. Run B (1172ae8) reproduced run A (876c18e) exactly on the same machine: all
352 per-task results (three limits, 118 tasks) and every family's S_N are identical.

Run A's leak-audit counts came from patch 0058. Patch 0060 changed two of the audit's rules, and run B
used them:
- zeros that only pad a printed number are not significant digits;
- a numeric match that chance explains on that number's grid is reported as `answer_lookup_chance`
  (info), not counted as a leak.

Run A flagged nine tasks. In run B no solver-facing file of any task contains the hidden law. One
high finding remains: a deliberate hint in the metadata of the surface-tension task (the derived
input `V_m = M_mol / rho_liq`, a sub-expression of the law). Ten tasks have matches that chance
explains. In electron-impact ionization the matched constant was 1e16, a unit conversion; since
patch 0062, numbers with a single significant digit are never treated as specific.
