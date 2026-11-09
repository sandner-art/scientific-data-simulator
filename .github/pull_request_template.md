## What and why

<!-- One or two sentences. Link the issue if there is one. -->

## Checks

- [ ] `pytest` passes (and `pytest ui/tests` if the UI changed)
- [ ] `sds examples check --run-quick` passes, if examples or the engine changed
- [ ] A test covers the change (for a bug fix: it fails without the fix)
- [ ] `CHANGELOG.md` updated under the unreleased version
- [ ] Docs updated (README, `examples/README.md`, `ui/README.md`, `docs/`), if behaviour changed

For a release, also follow `RELEASING.md` (versions in both `pyproject.toml` files, `CITATION.cff`, `docs/conf.py`).
