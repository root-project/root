# Release Process

A cppjit release is a git tag on main. The tag push builds the wheels,
and the PyPI workflow uploads what that build produced. Future releases
will extend this to more distribution channels like conda.

## Procedure

1. Open the release PR:
   - Set `__version__` in `python/cppjit/_version.py` to the release
     version.
   - Fill in `docs/ReleaseNotes.md`.
   - Add the `build-wheels` label to build the complete set of wheels
     on the PR.

2. Squash-merge the PR and tag the resulting commit on main:

   ```bash
   git tag -a vX.Y.Z -m "cppjit X.Y.Z" <commit>
   git push origin vX.Y.Z
   ```

   The tag push runs the Wheels workflow with every supported Python
   on every platform.

3. When that run succeeds, the PyPI workflow uploads its wheels and
   sdist. If the `pypi` environment has required reviewers, the
   upload waits for their approval. A failed run uploads nothing: fix
   the problem and re-tag, or, if only the upload failed, re-run it
   through the manual fallback below.

4. Create the GitHub release for the tag. The body is the finalized
   `docs/ReleaseNotes.md`. The attached artifacts are the uploaded
   wheels and sdist.

5. Announce the release on the compiler-research website and the
   relevant mailing lists.

6. Open the post-release PR. It sets `_version.py` to the next
   version with a `.dev0` suffix (e.g., `0.1.0a2.dev0` follows
   `0.1.0a1`) and resets `docs/ReleaseNotes.md`.

## Version and metadata

`python/cppjit/_version.py` is the source of truth for the version.
scikit-build-core reads it at build time through the
`[[tool.dynamic-metadata]]` entry in `pyproject.toml`. PyPI uses the
information in `pyproject.toml` to build the project page. The
versioning scheme follows PEP 440, and between releases the version
carries a `.dev0` suffix.

## Release notes

`docs/ReleaseNotes.md` collects notes for the release under
development. The release PR completes them and the post-release PR
resets them, so a git checkout always describes the next release.
The notes of a released version live on in its GitHub release.

PyPI shows no release notes. It renders the README that was built
into each version, and the `Changelog` URL in `[project.urls]` leads
from there to the GitHub releases page.

## The PyPI workflow

`.github/workflows/pypi.yml` runs when the Wheels run for a `v` tag
completes. On success it downloads that run's wheels and sdist and
uploads them to PyPI, so what lands there is exactly what CI built
and tested. The upload uses trusted publishing: the job authenticates
with a short-lived OIDC token, and the repository stores no
credential. GitHub runs the copy of `pypi.yml` on main, so a change
to the workflow takes effect once it merges.

The manual fallback is a `workflow_dispatch` of the same workflow
with the run id of a successful full Wheels run.

PyPI accepts each version once, and deleting an upload does not free
its number. A broken release is replaced by the next patch or
prerelease number.
