# Changelog

## V1.2.2 - 2026-09-23

- Select a supported Python on macOS, including versioned python.org and Homebrew installations available when opening setup from Finder.
- Stop before package installation if only an unsupported Python such as Apple's Python 3.9 is available.
- Preserve unsupported local environments before rebuilding them; keep student results in place and reject unsupported environments in the launcher.
- Skip install-time byte compilation of dependency templates and report the failed setup stage accurately.
- Add regression tests for Python selection, environment preservation, and launcher checks, plus a macOS Python 3.13 CI run.
- Clarify macOS installation and keep direct Windows, macOS, and source download links in the README.

## V1.2.1 - 2026-09-22

- Accept the English review in the manuals; remove long sentence dashes while retaining ranges and mathematical minus signs.
- Include physics-based figures with consistent printed text sizes, verified numbering and corrected flowchart arrows.
- Build and audit manual figures in a standalone software clone, without requiring the separate manuscript or historical asset folders.
- Keep offline setup offline when local wheels are supplied.
- Prepare clean source, Windows and macOS archives with SHA-256 manifests and executable macOS scripts, preserving previous releases and user results.
- Run the scientific figure tests in GitHub Actions through explicit development dependencies.

## V1.2.0 - 2026-09-21

- Preserve fractional positions in Gaussian rendering and normalize each localization to one count.
- Export crop-local, original-frame, and optional calibrated coordinates.
- Add frame-aware one-to-one truth matching with precision, recall, F1, crop/frame scoping, and explicit match radius.
- Apply optional frame-median background subtraction; skip blank frames and reject malformed input.
- Run reconstruction in a background worker, reduce frame storage, and protect existing output directories.
- Stage and validate user packages, preserve previous packages including user results, and verify SHA-256 manifests against source.
- Synchronize manuals, version metadata, and regression tests.

## V1.1.0 - 2026-05-20

- Added `apps/ClassRoomSTORM_Studio.py`, a PySide6 Studio launcher with a live super-resolution intro background.
- Added workflow cards for opening the Virtual Experiment and SR Reconstruction apps from one front door.
- Added Skip intro and Replay animation controls for the animated launch screen.
- Kept the V1 virtual experiment and reconstruction workflows available as direct entry points.

## V1.0.0 - 2026-05-13

- Added public app entry points:
  - `apps/ClassRoomSTORM_VirtualExperiment.py`
  - `apps/ClassRoomSTORM_Reconstruction.py`
- Standardized output folder names with `ClassRoomSTORM_*` prefixes.
- Added release documentation and citation files.
- Added result browser support for image/SVG zoom and scrollable document viewing.
- Added teaching pipeline report, flowchart, and equation map outputs.
