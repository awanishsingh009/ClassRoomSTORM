#!/bin/bash
cd "$(dirname "$0")"

fail() {
    echo
    echo "Setup did not finish."
    echo "This can happen on restricted networks, proxy networks, or machines"
    echo "where Python cannot verify PyPI SSL certificates."
    echo "The local environment has been kept so setup can be retried."
    echo "The launcher checks required imports before starting."
    echo
    echo "See README_FIRST.txt for troubleshooting notes."
    read -p "Press Return to close."
    exit 1
}

echo
echo "ClassRoomSTORM Studio V1.2 - macOS setup"
echo "========================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 was not found."
    echo "Please install Python 3.10 or newer, then run this script again."
    read -p "Press Return to close."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating local Python environment..."
    python3 -m venv .venv || fail
fi

echo "Installing ClassRoomSTORM requirements..."
if ls wheels/*.whl >/dev/null 2>&1; then
    echo "Found a local wheels folder - installing offline."
    ".venv/bin/python" -m pip install --no-index --find-links wheels -r requirements.txt || fail
else
    echo "Upgrading pip..."
    ".venv/bin/python" -m pip install --upgrade pip || fail
    ".venv/bin/python" -m pip install -r requirements.txt || fail
fi

echo
echo "Setup completed."
echo "You can now run launch_studio_macos.command."
read -p "Press Return to close."
