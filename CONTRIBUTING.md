# Contributing

Thank you for helping. Bug reports, new models, examples and fixes are all welcome.

## Set up

```bash
bash scripts/install.sh dev        # Windows: scripts\install.cmd dev   (add the UI: pip install -e ./ui)
source .venv/bin/activate          # Windows: .venv\Scripts\activate
sds doctor
```

## Before you send a change

```bash
pytest                             # the simulator (about 70 s)
pytest ui/tests                    # the UI, if you touched it (needs: pip install -e ./ui)
sds examples check --run-quick     # every example in its quick mode (about 3 min)
```

CI runs all three on Linux, Windows and macOS.

## Adding things

- **A model:**
  - Put an `ExperimentLogic` package in `experiments/<name>/`.
  - Give it a fidelity level and a note: `reference` only with tests that check it against known
    results, `audit` for a deliberately weak model with its flaw documented and pinned by a test.
  - Check it with `simulator.checker.check_model` (see "Checking a model" in the README).
- **An example:** a folder in `examples/` with `run_experiment.py` and an `example.yaml` (see
  `examples/README.md`). It then appears in `sds examples list` and in the UI.
- **The UI** lives in `ui/`, a separate package. It may use only the simulator's public
  functions; the simulator must never import it (a test checks both rules).

## Style

- Plain, short docstrings that say what a function does and what it returns.
- Seeded randomness only through the engine's generators, never global `np.random`.
- Every change that fixes a bug comes with a test that fails without the fix.
- Update `CHANGELOG.md` under the unreleased version.

Releases follow `RELEASING.md`.
