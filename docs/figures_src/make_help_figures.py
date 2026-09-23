"""Compatibility entry point for the consolidated, physics-reviewed figure build.

Original generators are preserved in the pre-revision archive. All current
numerical and TikZ sources are rebuilt together to prevent style/data drift.
"""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    raise SystemExit(subprocess.call([sys.executable, str(root / "tools/build_figures.py")], cwd=root))
