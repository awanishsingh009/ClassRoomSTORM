# ClassRoomSTORM User Guide

## Before You Start

ClassRoomSTORM V1.1 is a Python desktop application. Install these before using it:

- Python 3.10 or newer.
- Standard Python packages from `requirements.txt`.
- Optional: Graphviz for cleaner pipeline SVG diagrams.
- Optional: NVIDIA CUDA PyTorch for experimental GPU acceleration.

The normal student installation does not require Graphviz or GPU packages.

## Install Standard CPU Version

Open a terminal in the `ClassRoomSTORM_V1` folder.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the required CPU packages:

```powershell
python -m pip install -r requirements.txt
```

The standard requirements are:

```text
numpy
opencv-python
matplotlib
Pillow
PySide6
```

## Check The Installation

Run these commands from `ClassRoomSTORM_V1`:

```powershell
python apps\ClassRoomSTORM_Studio.py --help
python apps\ClassRoomSTORM_VirtualExperiment.py --help
python apps\ClassRoomSTORM_Reconstruction.py --help
```

If all commands show help text, the installation is ready.

## Optional Graphviz For Pipeline Figures

The reconstruction app can generate teaching flowcharts and equation maps without Graphviz, but Graphviz gives cleaner SVG diagrams. Install it only if you want the higher-quality Graphviz-rendered pipeline figures.

Windows:

```powershell
winget install Graphviz.Graphviz
```

macOS:

```bash
brew install graphviz
```

Ubuntu/Debian:

```bash
sudo apt install graphviz
```

After installation, restart the terminal and check:

```powershell
dot -V
```

If `dot -V` works, ClassRoomSTORM will use Graphviz automatically.

## Optional NVIDIA GPU Acceleration

GPU acceleration is experimental and only supported for NVIDIA CUDA in V1. Install it after the standard CPU requirements:

```powershell
python -m pip install -r requirements-gpu.txt
```

If CUDA is unavailable, reconstruction records the fallback in `summary.json` and continues on CPU. AMD GPU, Intel GPU, and Apple Silicon GPU acceleration are not implemented in V1.

## Launch

Recommended Studio launcher:

```powershell
python apps\ClassRoomSTORM_Studio.py
```

Direct workflow launch:

```powershell
python apps\ClassRoomSTORM_VirtualExperiment.py
python apps\ClassRoomSTORM_Reconstruction.py
```

## Virtual Experiment Workflow

1. Choose a pattern: text, dots, or logo.
2. Adjust sigma, frames, blinkers per frame, brightness, background, and noise.
3. Preview the source mask and a sample blinking frame.
4. Export the dataset.
5. Load the generated `blinking_video.mp4` in the reconstruction app.

Virtual output files:

- `blinking_video.mp4`: video used for reconstruction.
- `truth.csv`: known source positions for comparison after reconstruction only.
- `metadata.json`: generation settings.
- `preview_frame.png`: sample blinking frame.
- `pattern_mask.png`: source mask.

## Reconstruction Workflow

1. Load a real or virtual blinking video.
2. Inspect the first frame.
3. Optionally crop the field of view.
4. Choose blinker mode: auto, single, or multiple.
5. Choose processing backend.
6. Run reconstruction.
7. Inspect results in the result browser.

Reconstruction output files:

- `raw_mean.png`: mean image from processed frames.
- `superres.png`: rendered super-resolution image.
- `raw_vs_superres.png`: comparison figure.
- `localizations.csv`: localization table.
- `summary.json`: run settings, timings, backend, and counts.
- `plot_*.png`: diagnostic and teaching plots.
- `pipeline_report.md` / `pipeline_report.html`: teaching explanation.
- `pipeline_flowchart.svg` and `pipeline_equation_map.svg`: pipeline figures.
- `extra_files/`: DOT and PNG copies of pipeline figures.

## Instructor Notes

Short course: use the virtual experiment first, then reconstruct the exported MP4. Longer course: add a real blinking video from a classroom optical setup, LED source, microscopy-like setup, or other blinking data source.

Learning goals:

- Compare raw mean images with localization-based reconstruction.
- Understand why truth data is hidden until after reconstruction.
- Explore how sigma, noise, density, and overlap affect reconstruction.
- Treat physical calibration as optional; the core workflow is pixel-first.

## Troubleshooting

If video loading fails, check that the file is an MP4, AVI, or MOV readable by OpenCV. If no localizations are found, inspect the frame, try multiple mode, and check brightness/noise. If reconstruction is slow, reduce max frames for a test run or use Auto CPU. For truth comparison, make sure `truth.csv` belongs to the same virtual experiment folder and was not used before reconstruction.

## V1.2 controls

Manual calibration applies the entered unit-per-pixel scale to original-frame CSV coordinates. Frame-median background subtraction and threshold quantile are available on Preprocess. Blank frames are skipped automatically. Truth match radius controls the one-to-one comparison; summary.json reports precision, recall, F1 and matched errors. The window remains responsive during reconstruction. Wait for export to finish and use a new empty output folder for each run.
