#!/bin/bash
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
    PYEXE=".venv/bin/python"
else
    echo "Local environment was not found. Trying system python3."
    PYEXE="python3"
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
