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
   - set the version (no `.dev0`) in `pyproject.toml`, `CITATION.cff` (`version`,
     `date-released`) and `docs/conf.py`, and the same version in `ui/pyproject.toml` (also its
     `scientific-data-simulator>=` floor) and `ui/sds_ui/__init__.py`; the tests check they agree;
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

## Optional: PyPI

Before uploading, decide how examples ship: they are not part of the installed package. The
`sds examples` commands find them in a source checkout, or through `SDS_EXAMPLES`.

Publish with PyPI trusted publishing from a GitHub Actions workflow triggered by the release.
Then `pip install scientific-data-simulator` works without a clone.

## Citing in a paper

Cite the version DOI of the release that produced your results, not the concept DOI. Each
result folder records the simulator version in `experiment_record.json`, and example runs
record it in `run.json`.
