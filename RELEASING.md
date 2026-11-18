# Releasing

`main` holds released code. Development happens on a branch and reaches `main` through a pull
request when a release is ready. Version tags (`vX.Y.Z`) are made only on `main`. The changelog
lists every version; versions 0.2–0.6 were development steps, not separate releases.

## One-time setup

1. **Mark the prototype.** Tag the original commit (`370af26`, March 2025) as `v0.1.0`. Publish it as
   a GitHub release marked *pre-release*, titled "0.1.0 – prototype (superseded)", and do this
   *before* step 3, so Zenodo does not archive the prototype:

   ```bash
   git tag -a v0.1.0 370af26 -m "0.1.0: prototype (superseded)"
   git push origin v0.1.0
   ```

2. **Bring the development branch in.** Push the branch that carries the patches, open a pull
   request into `main`, and wait for CI to pass on Linux, Windows and macOS. Merge with a merge
   commit; that keeps one commit per version.
3. **Connect Zenodo.** Log in to Zenodo with GitHub and switch the repository on under
   *GitHub*. Each published GitHub release made after that is archived with a version DOI. All
   versions share one concept DOI, which always resolves to the latest version. Records cannot be
   deleted, so try it first on the sandbox (sandbox.zenodo.org).

   Zenodo takes its metadata from `CITATION.cff`. Do not add a `.zenodo.json` unless you need
   fields CFF lacks, such as grants or communities. When both exist Zenodo uses only
   `.zenodo.json`, and the two would drift apart.

## Each release

1. On the development branch:
   - set the version (no `.dev0`; a release candidate is e.g. `0.7.0rc2`) in `pyproject.toml`,
     `CITATION.cff` (`version`, `date-released`), `docs/conf.py` and the README's citation line
     (also its year), and the same version in `ui/pyproject.toml` (also its
     `scidatasim>=` floor) and `ui/sds_ui/__init__.py`; the tests check they agree;
   - date the changelog section;
   - run `pytest`, `pytest ui/tests` and `sds examples check --run-quick`.
2. Open a pull request into `main`. CI must pass on all three systems, including the
   install-script job.
3. For a first release, or a large one, tag a release candidate on `main` first, for example
   `v0.7.0rc1`, as a GitHub *pre-release*. Have a few people install it with
   `scripts/install.*` on their own machines, fix what they find, then tag the final version.
4. Tag `vX.Y.Z` on `main` and publish a GitHub release whose notes are that changelog section.
   Zenodo archives it.
5. Back on the development branch:
   - add the version DOI to `CITATION.cff` (`identifiers`) and the concept DOI to the README badge;
   - bump to the next `.dev0` version;
   - merge `main` back in so the branches do not diverge.

## PyPI

`.github/workflows/publish.yml` builds both packages (`scidatasim` and
`scidatasim-ui`; the repository keeps its
longer name), checks them with `twine check --strict`, installs the two wheels in
a clean environment and runs `sds doctor`, then uploads them with *trusted publishing*: GitHub
proves to PyPI which workflow is uploading, so no token is stored anywhere.

One-time setup:

1. Create accounts on [pypi.org](https://pypi.org) and [test.pypi.org](https://test.pypi.org)
   (two-factor authentication is required).
2. In the GitHub repository, *Settings* > *Environments*: create `pypi` and `testpypi`. Give `pypi`
   a required reviewer (yourself), so nothing reaches PyPI without your click.
3. On each site, under *Account* > *Publishing*, add a **pending publisher** for `scidatasim`:
   owner `sandner-art`, repository `scientific-data-simulator`, workflow `publish.yml`,
   environment `pypi` (on test.pypi.org: `testpypi`). The project name must be the `name` in
   `pyproject.toml` (`ui/pyproject.toml` for the UI), or the upload is refused. A pending
   publisher does not reserve the name; the first upload creates the project.
4. PyPI takes only one pending publisher per owner, repository, workflow and environment, so
   `scidatasim-ui` is added after `scidatasim` exists. The first upload to each site therefore
   takes two runs: the first run creates `scidatasim` and fails on `scidatasim-ui`; then add the
   pending publisher for `scidatasim-ui` (same details) and use *Re-run failed jobs*. Both
   upload jobs skip files that are already there, so the re-run uploads only the UI. From then on
   one run publishes both.

Each release:

1. **Dry run.** *Actions* > *publish* > *Run workflow* on the branch to release uploads to
   TestPyPI. Install from there in a fresh environment and run `sds doctor`:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
       "scidatasim==X.Y.Z"
   ```

2. **Publish.** Publishing the GitHub release (step 4 above) runs the workflow again and, after
   your approval, uploads to PyPI. The tag must be `v` plus the version in `pyproject.toml`, and a
   `.dev` version is refused. A release candidate (`0.7.0rc2`) is a pre-release on PyPI too: pip
   installs it only when asked for it (`pip install --pre ...` or `==0.7.0rc2`).

PyPI never accepts a version twice, and a published version cannot be replaced, only yanked; fix
problems in the next version. Examples are not part of the package: `sds examples` finds them in a
source checkout, or through `SDS_EXAMPLES`.

## Datasets

Example 13 (`examples/example_13_law_release`) builds a frozen law-discovery dataset: secret seeds,
answers withheld, reference baselines and a canary in the dataset card, a leak audit that must
pass, and checksums. Its README describes the upload to the Hugging Face Hub and how to get a
DOI. Release a dataset after the software version that built it, and name that version in the
dataset card; keep the private folder, which is the only answer key.

## Citing in a paper

Cite the version DOI of the release that produced your results, not the concept DOI. Each
result folder records the simulator version in `experiment_record.json`, and example runs
record it in `run.json`.
