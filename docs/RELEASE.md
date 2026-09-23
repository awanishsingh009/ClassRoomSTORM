# Preparing a GitHub update

The maintained repository is the `ClassRoomSTORM_V1` directory. Version 1.2.1
contains the reviewed manuals, scientific figures and release-preparation fixes.
The parent research workspace and the separate manuscript are not required to run
or test this software. Keep historical backups and student results locally.

## Verify the source

Use Python 3.11 and a project environment. From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/validate_release.py --output ../release-validation-new
```

The validation output must be new or empty. The smoke check generates a seeded
virtual video, reconstructs it, compares with truth, and processes 40 frames of
the included experimental video. It does not establish experimental accuracy.
The GitHub workflow runs tests on Windows, Linux and macOS after upload; a local
Windows pass does not establish that those hosted runs have passed.

## Rebuild documentation when it changes

The PDFs are included, so users do not need a TeX installation. Developers need
LaTeX, dvipng and Poppler on PATH to regenerate figures. The optional figure
environment needs Python 3.11 or newer:

```bash
python -m pip install -r requirements-figures.txt
python tools/build_figures.py
python tools/build_docs.py
python tools/audit_figure_typography.py
```

A standalone clone builds the two manuals and their 20 figure entries. When the
research workspace also contains `../manuscript_writing/ClassRoomSTORM_AJP.tex`,
the tools include the companion manuscript's figures and locations. Compile that
manuscript before running the full typography audit. The apparatus reference image
is included with the editable figure sources; the historical asset tree is not a
build dependency. Check rendered pages after changing text or artwork.

## Create the upload bundle

```bash
python tools/prepare_github_release.py --output ../releases/ClassRoomSTORM-1.2.1-ready
```

Choose a new directory each time. This creates a clean `ClassRoomSTORM_Source/`
folder, Windows and macOS packages, three ZIPs, manifests, checksums and release
notes. It preserves existing releases. The source snapshot contains tests,
documentation, figure data, editable TikZ/Python sources and GitHub Actions.
It excludes environments, student results, Git metadata, backups and generated
build intermediates. macOS scripts retain executable permissions in the ZIPs.

```bash
python tools/prepare_github_release.py --validate-source ../releases/ClassRoomSTORM-1.2.1-ready/ClassRoomSTORM_Source
python tools/build_user_packages.py --validate --output-dir ../releases/ClassRoomSTORM-1.2.1-ready
```

To refresh the adjacent classroom packages only, run `python tools/build_user_packages.py`.
The builder archives existing package folders, including any student results.
The older `--zip` option refuses to replace an existing ZIP of the same version.

## Publish later

Review the diff in the maintained repository or copy the exported source into a
fresh clone of your GitHub repository. Keep that clone's `.git` directory and
history. Do not copy the parent research workspace or the user packages into the
source repository. Review existing tracked `backups/` entries separately if you
use the maintained checkout; `.gitignore` does not untrack historical commits.

Check the repository URL with `git remote -v`, create or choose the intended
branch, review `git diff` and `git status`, then commit and push when ready.
Wait for the GitHub checks to pass before tagging `v1.2.1` or publishing a release.
Attach the Windows/macOS ZIPs and `SHA256SUMS.txt` as release downloads. The
preparation tool performs no commit, remote setup, push, tag or publication.
