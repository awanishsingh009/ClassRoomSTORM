#!/bin/bash
cd "$(dirname "$0")" || exit 1

if [ -x ".venv/bin/python" ]; then
    PYEXE=".venv/bin/python"
else
    echo "Please run setup_macos.command first."
    read -r -p "Press Return to close."
    exit 1
fi

if ! "$PYEXE" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1; then
    echo "The local Python environment needs to be replaced."
    echo "Run setup_macos.command to select Python 3.10-3.13 and preserve the old environment."
    read -r -p "Press Return to close."
    exit 1
fi

if ! "$PYEXE" -c "import numpy, cv2, matplotlib, PIL, PySide6" >/dev/null 2>&1; then
    echo
    echo "Required Python packages are not installed."
    echo "Please run setup_macos.command first, then try again."
    read -p "Press Return to close."
    exit 1
fi

"$PYEXE" apps/ClassRoomSTORM_Studio.py
status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ClassRoomSTORM did not launch successfully."
    echo "Try running setup_macos.command first."
    read -p "Press Return to close."
fi
exit "$status"
