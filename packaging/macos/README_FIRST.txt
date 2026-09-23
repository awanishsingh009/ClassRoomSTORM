ClassRoomSTORM Studio V1.2 - macOS User Package

This folder contains everything needed to run ClassRoomSTORM Studio on macOS.

------------------------------------------------------------------
Quick start
------------------------------------------------------------------
1. Install Python 3.10 or newer from https://www.python.org/
   (On Apple Silicon Macs, use the standard macOS universal2 installer.)
2. Open Terminal in this folder and run, once:
      chmod +x setup_macos.command launch_studio_macos.command
3. Double-click setup_macos.command and wait for "Setup completed".
   The first time, macOS may block the file because it was downloaded:
   right-click setup_macos.command and choose "Open", then confirm.
4. Double-click launch_studio_macos.command (right-click - Open the first time).
5. In the Studio launcher, use:
   - Virtual Experiment   create blinking data from a logo, letter, or pattern
   - SR Reconstruction    reconstruct a recorded or simulated blinking video
   - Help                 open this user guide as a PDF
   - Theory               open the physics companion document

------------------------------------------------------------------
What setup needs
------------------------------------------------------------------
- Python 3.10 or newer (3.11 recommended).
- An internet connection the first time: setup downloads about 250-350 MB of
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
- "python3 was not found": install Python 3.10 or newer, then run
  setup_macos.command again.
- A macOS security warning when you double-click a .command file: right-click
  the file and choose "Open", or remove the download quarantine with
      xattr -d com.apple.quarantine setup_macos.command launch_studio_macos.command
- SSL / certificate / proxy errors (a "CERTIFICATE_VERIFY_FAILED" message):
  your network is blocking or intercepting access to PyPI. Try an unrestricted
  network, ask your IT support how pip should reach PyPI, or use a package that
  includes an offline "wheels" folder.
- If setup fails partway, it preserves the environment for a retry. The
  launcher checks required imports before starting. Fix the problem and run
  setup_macos.command again.
- "Required Python packages are not installed" from the launcher means setup did
  not complete - run setup_macos.command first.

Optional GPU packages are not installed by default. The macOS package uses CPU
processing; CUDA GPU acceleration is not available on macOS. Keep Virtual
Experiment on Auto CPU or CPU parallel, and keep SR Reconstruction on Serial or
Auto CPU.
