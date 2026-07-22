ClassRoomSTORM Studio V1.1 - Windows User Package

This folder contains everything needed to run ClassRoomSTORM Studio on Windows.

------------------------------------------------------------------
Quick start
------------------------------------------------------------------
1. Double-click setup_windows.bat and wait until it prints "Setup completed".
   The setup script looks for Python 3.11, 3.12, or 3.13 automatically. If none
   is found, it can offer to install Python 3.11 with winget or open the official
   Python download page.
2. Double-click launch_studio_windows.bat.
3. In the Studio launcher, use:
   - Virtual Experiment   create blinking data from a logo, letter, or pattern
   - SR Reconstruction    reconstruct a recorded or simulated blinking video
   - Help                 open this user guide as a PDF
   - Theory               open the physics companion document

------------------------------------------------------------------
What setup needs
------------------------------------------------------------------
- Python 3.11, 3.12, or 3.13. Python 3.11 is recommended.
- An internet connection the first time: setup downloads about 250-350 MB of
  packages (numpy, opencv-python, matplotlib, Pillow, PySide6) from PyPI.
- If this folder contains a "wheels" folder, setup installs from it offline and
  no internet connection is needed.

------------------------------------------------------------------
Included files
------------------------------------------------------------------
- apps\                          application programs
- classroomstorm_core\           shared program code
- assets\                        sample video and logo
- docs\ClassRoomSTORM_Help.pdf   this user guide
- docs\ClassRoomSTORM_Theory.pdf the physics companion document
- examples\                      space for your own example datasets
- setup_windows.bat, launch_studio_windows.bat
- requirements.txt, requirements-gpu.txt, LICENSE, CITATION.cff

You can open the two PDFs in docs\ directly at any time, without launching
the application.

Sample video for SR Reconstruction:
  assets\experimental_led_matrix_video_2023-09-07.mp4

------------------------------------------------------------------
If setup fails
------------------------------------------------------------------
- "No supported Python was found": choose the setup script option to install
  Python 3.11 with winget, or open the official Python download page. Then run
  setup_windows.bat again.
- If typing "python" opens the Microsoft Store or uses Python 3.14, disable the
  python.exe and python3.exe aliases in:
  Settings > Apps > Advanced app settings > App execution aliases
  The setup script will still prefer a real Python 3.11, 3.12, or 3.13 found
  through the Python launcher (`py`) when available.
- SSL / certificate / proxy errors (a "CERTIFICATE_VERIFY_FAILED" message):
  your network is blocking or intercepting access to PyPI. Try an unrestricted
  network, ask your IT support how pip should reach PyPI, or use a package that
  includes an offline "wheels" folder.
- If setup fails partway, it removes the incomplete environment automatically,
  so the launcher will not start from a broken install. Fix the problem and run
  setup_windows.bat again.
- "Required Python packages are not installed" from the launcher means setup did
  not complete - run setup_windows.bat first.

Optional GPU packages are not installed by default; the standard package uses
CPU processing. On NVIDIA CUDA machines, install requirements-gpu.txt to try
experimental GPU CUDA frame generation in Virtual Experiment and GPU backends
in SR Reconstruction. If CUDA is unavailable, the app falls back to CPU and
records the reason in the output metadata or summary file.
