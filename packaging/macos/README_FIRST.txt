ClassRoomSTORM Studio V1.2 - macOS User Package

This folder contains everything needed to run ClassRoomSTORM Studio on macOS.

------------------------------------------------------------------
Quick start
------------------------------------------------------------------
1. Install Python 3.11, 3.12, or 3.13 from
   https://www.python.org/downloads/macos/
   (On Apple Silicon Macs, use the standard macOS universal2 installer.)
2. Open Terminal in this folder and run, once:
      chmod +x setup_macos.command launch_studio_macos.command
3. Double-click setup_macos.command and wait for "Setup completed".
   If macOS cannot verify the developer, open System Settings > Privacy &
   Security after attempting to run it, then choose "Open Anyway" for this file.
4. Double-click launch_studio_macos.command. Approve this file separately in
   Privacy & Security if prompted. Use the files from the official release.
5. In the Studio launcher, use:
   - Virtual Experiment   create blinking data from a logo, letter, or pattern
   - SR Reconstruction    reconstruct a recorded or simulated blinking video
   - Help                 open this user guide as a PDF
   - Theory               open the physics companion document

------------------------------------------------------------------
What setup needs
------------------------------------------------------------------
- Python 3.10-3.13 (3.11 recommended). Apple's developer-tools Python 3.9 is
  not supported. Setup checks versioned Python installations automatically,
  including python.org and common Homebrew locations when launched from Finder.
- An internet connection the first time: setup downloads several hundred MB of
  packages (numpy, opencv-python, matplotlib, Pillow, PySide6) from PyPI.
- If this folder contains a "wheels" folder, setup installs from it offline and
  no internet connection is needed.

------------------------------------------------------------------
Included files
------------------------------------------------------------------
- apps/                          application programs
- classroomstorm_core/           shared program code
- assets/                        sample video and logo
- docs/ClassRoomSTORM_Help.pdf   this user guide
- docs/ClassRoomSTORM_Theory.pdf the physics companion document
- examples/                      space for your own example datasets
- setup_macos.command, launch_studio_macos.command
- requirements.txt, requirements-gpu.txt, LICENSE, CITATION.cff

You can open the two PDFs in docs/ directly at any time, without launching
the application.

Sample video for SR Reconstruction:
  assets/experimental_led_matrix_video_2023-09-07.mp4

------------------------------------------------------------------
If setup fails
------------------------------------------------------------------
- "No supported Python was found": install Python 3.11, 3.12, or 3.13, then run
  setup_macos.command again. Installing Python does not replace an existing
  virtual environment; this setup preserves an unsupported environment in a
  .venv-backup-* folder before creating a new one. Student results stay in place.
- If an older setup failed with Python 3.9, PySide6 __init__.tmpl.py, or
  "encode() argument 'encoding' must be str, not None", use the v1.2.2 or newer
  package and install a supported Python as above. No system Python removal is
  needed. Setup skips byte compilation of package templates during installation.
- "Developer cannot be verified": try opening the .command file, then approve
  that file in System Settings > Privacy & Security > Open Anyway.
- "Permission denied": open Terminal in this package folder and run
      chmod u+x setup_macos.command launch_studio_macos.command
  Then open setup_macos.command again.
- SSL / certificate / proxy errors (a "CERTIFICATE_VERIFY_FAILED" message):
  your network is blocking or intercepting access to PyPI. Try an unrestricted
  network, ask your IT support how pip should reach PyPI, or use a package that
  includes an offline "wheels" folder.
- If setup fails partway, it preserves the environment for a retry. The
  launcher checks required imports before starting. Fix the problem and run
  setup_macos.command again.
- "Required Python packages are not installed" from the launcher means setup did
  not complete - run setup_macos.command first.

Advanced: CLASSROOMSTORM_PYTHON can specify a supported interpreter when
creating a new environment. A valid existing environment is reused.

Optional GPU packages are not installed by default. The macOS package uses CPU
processing; CUDA GPU acceleration is not available on macOS. Keep Virtual
Experiment on Auto CPU or CPU parallel, and keep SR Reconstruction on Serial or
Auto CPU.
