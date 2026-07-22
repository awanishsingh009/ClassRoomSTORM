# ClassRoomSTORM Studio V1.1

[![Tests](https://github.com/awanishsingh009/ClassRoomSTORM/actions/workflows/tests.yml/badge.svg)](https://github.com/awanishsingh009/ClassRoomSTORM/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ClassRoomSTORM is a cross-platform Python teaching toolkit for stochastic blinking, localization microscopy, and super-resolution reconstruction. It connects an interactive virtual experiment with an independent video-reconstruction workflow so students can explore how blinking statistics, noise, emitter density, and localization create a reconstructed image.

![ClassRoomSTORM Studio launcher](docs/figures/help/H2_launcher_screenshot.png)

## Workflows

- **Virtual Experiment:** generate a blinking MP4 from text, dots, or a logo with configurable point-spread width, photon count, background, read noise, emitter density, and spacing.
- **SR Reconstruction:** inspect and crop real or simulated videos, localize single or multiple bright regions, render a reconstruction, and export quantitative and teaching outputs.
- **Truth-based validation:** compare reconstructed localizations with simulation truth only after reconstruction. Truth coordinates are never used by the reconstruction algorithm.
- **Teaching reports:** export figures, localization tables, JSON metadata, pipeline reports, and linked line-profile visualizations.
- **Processing backends:** use serial or parallel CPU processing, with optional experimental NVIDIA CUDA acceleration and documented CPU fallback.

## Scope

ClassRoomSTORM is deliberately transparent educational software. The V1.1 reconstruction uses percentile thresholding, connected components for multiple emitters, and intensity-weighted centroid localization. It does not replace research-grade SMLM packages with Gaussian/MLE fitting, drift correction, uncertainty estimation, or camera-specific calibration.

## Quick Start

Python 3.11 is recommended. Python 3.10-3.13 is supported.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python apps/ClassRoomSTORM_Studio.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. On macOS or Linux, use `source .venv/bin/activate`.

The `packaging/windows` and `packaging/macos` folders contain setup and launch scripts used to build the user packages.

## Command-Line Example

Generate a virtual ABBE experiment:

```bash
python apps/ClassRoomSTORM_VirtualExperiment.py \
  --generate-demo --pattern text --text ABBE --frames 100 \
  --out results/virtual_experiments/ABBE_demo
```

Reconstruct the generated video and compare with truth afterward:

```bash
python apps/ClassRoomSTORM_Reconstruction.py \
  --run-video results/virtual_experiments/ABBE_demo/blinking_video.mp4 \
  --out results/reconstructions/ABBE_demo --blinker-mode auto --compare-truth
```

## Outputs

Virtual experiments produce `blinking_video.mp4`, `truth.csv`, `metadata.json`, `preview_frame.png`, and `pattern_mask.png`.

Reconstructions produce `localizations.csv`, `summary.json`, NumPy arrays, rendered PNGs, diagnostic plots, and HTML/Markdown/SVG pipeline reports.

## Documentation

- [User Help](docs/ClassRoomSTORM_Help.pdf): installation, controls, workflows, outputs, and troubleshooting.
- [Theory Notes](docs/ClassRoomSTORM_Theory.pdf): diffraction, PSFs, blinking, noise, centroid localization, precision, and limitations.
- [User Guide](docs/ClassRoomSTORM_User_Guide.md): concise installation and workflow reference.
- [Technical Notes](docs/ClassRoomSTORM_Technical_Notes.md): algorithm, backend, and design scope.

## Tests

```bash
python -m pytest -q
```

The test suite covers simulation, reconstruction, CPU/GPU fallback behavior, frame-aware truth comparison, plots, reports, result catalogs, and the Studio launcher.

## Author

Dr. Awanish Pratap Singh<br>
Institute of Biomedical Optics, University of Lübeck

## Citation And License

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). ClassRoomSTORM is released under the [MIT License](LICENSE).
