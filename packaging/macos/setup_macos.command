#!/bin/bash
cd "$(dirname "$0")" || exit 1

fail() {
    echo
    echo "Setup did not finish."
    echo "$1"
    echo "Read the error above for details. Existing files are kept for a retry."
    echo
    echo "See README_FIRST.txt for troubleshooting notes."
    read -r -p "Press Return to close."
    exit 1
}

supported_python() {
    "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1
}

find_python() {
    if [ -n "${CLASSROOMSTORM_PYTHON:-}" ]; then
        if supported_python "$CLASSROOMSTORM_PYTHON"; then
            STORM_PYTHON="$CLASSROOMSTORM_PYTHON"
            return 0
        fi
        echo "CLASSROOMSTORM_PYTHON must point to Python 3.10-3.13."
        return 1
    fi
    # Finder may not inherit the PATH configured in the user's shell.
    local version candidate
    for version in 3.11 3.12 3.13 3.10; do
        for candidate in "python$version" \
            "/Library/Frameworks/Python.framework/Versions/$version/bin/python$version" \
            "/opt/homebrew/bin/python$version" \
            "/opt/homebrew/opt/python@$version/bin/python$version" \
            "/usr/local/bin/python$version" \
            "/usr/local/opt/python@$version/bin/python$version"; do
            if supported_python "$candidate"; then
                STORM_PYTHON="$candidate"
                return 0
            fi
        done
    done
    if supported_python python3; then
        STORM_PYTHON=python3
        return 0
    fi
    return 1
}

echo
echo "ClassRoomSTORM Studio V1.2 - macOS setup"
echo "========================================"
echo

if ! supported_python ".venv/bin/python"; then
    if ! find_python; then
        echo "No supported Python was found."
        echo "Install Python 3.11, 3.12, or 3.13 from https://www.python.org/downloads/macos/"
        echo "Then run this setup again. Apple's Python 3.9 is not supported."
        fail "Setup stopped before installing packages."
    fi
    echo "Using Python: $STORM_PYTHON"
    "$STORM_PYTHON" --version
    if [ -e .venv ] || [ -L .venv ]; then
        STORM_BACKUP=".venv-backup-$(date +%Y%m%d-%H%M%S)-$$"
        if [ -e "$STORM_BACKUP" ] || [ -L "$STORM_BACKUP" ]; then
            fail "Environment backup already exists: $STORM_BACKUP"
        fi
        mv .venv "$STORM_BACKUP" || fail "Could not preserve the old environment."
        echo "Previous environment preserved in $STORM_BACKUP"
    fi
    echo "Creating local Python environment..."
    "$STORM_PYTHON" -m venv .venv || fail "Could not create the Python environment."
fi

echo "Package environment:"
".venv/bin/python" --version
echo "Installing ClassRoomSTORM requirements..."
# PySide6 ships Android template files that are not standalone Python modules.
# Skip install-time byte compilation; imported runtime modules compile normally.
if ls wheels/*.whl >/dev/null 2>&1; then
    echo "Found a local wheels folder - installing offline."
    ".venv/bin/python" -m pip install --no-compile --no-index --find-links wheels -r requirements.txt || fail "Could not install packages from the local wheels folder."
else
    echo "Upgrading pip..."
    ".venv/bin/python" -m pip install --upgrade pip || fail "Could not upgrade pip."
    ".venv/bin/python" -m pip install --no-compile -r requirements.txt || fail "Could not install the application packages."
fi

echo
echo "Setup completed."
echo "You can now run launch_studio_macos.command."
read -r -p "Press Return to close."
