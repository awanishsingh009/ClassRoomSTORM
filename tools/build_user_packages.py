#!/usr/bin/env python3
"""
build_user_packages.py -- repeatable builder for the ClassRoomSTORM V1 user packages.

Creates clean standalone Windows and macOS user packages from this development
repository. It copies only approved runtime files, excludes every developer file
(.git, tests, results, docs/latex, figures_src, raw .tex, LaTeX build artefacts,
backups, __pycache__, audit reports), validates the result, and writes a manifest.

Usage (run from anywhere):
    python tools/build_user_packages.py            build both packages, validate
    python tools/build_user_packages.py --validate validate existing packages only

Output folders are created next to this repository:
    ../ClassRoomSTORM_V1_User_windows
    ../ClassRoomSTORM_V1_User_macos
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO.parent

TARGETS = {
    "windows": "ClassRoomSTORM_V1_User_windows",
    "macos":   "ClassRoomSTORM_V1_User_macos",
}

# Runtime folders copied verbatim, minus the excludes below.
RUNTIME_DIRS = ["apps", "classroomstorm_core", "assets", "examples"]

# Top-level files copied as-is.
ROOT_FILES = ["LICENSE", "CITATION.cff", "requirements.txt", "requirements-gpu.txt"]

# Only the compiled PDFs are shipped from docs/ -- never the LaTeX sources.
DOC_PDFS = ["ClassRoomSTORM_Help.pdf", "ClassRoomSTORM_Theory.pdf"]

# Per-platform files taken from packaging/<platform>/.
PLATFORM_FILES = {
    "windows": ["setup_windows.bat", "launch_studio_windows.bat", "README_FIRST.txt"],
    "macos":   ["setup_macos.command", "launch_studio_macos.command", "README_FIRST.txt"],
}

EXCLUDE_DIRS = {"__pycache__", ".git", ".worktrees", ".pytest_cache",
                "build", "dist", ".idea", ".vscode", ".mypy_cache", "latex",
                "figures_src"}
EXCLUDE_FILE_SUFFIXES = (".pyc", ".pyo", ".bak", ".orig", ".tmp", ".log",
                         ".aux", ".toc", ".synctex.gz")
EXCLUDE_FILE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


def _excluded_file(name: str) -> bool:
    low = name.lower()
    if name in EXCLUDE_FILE_NAMES:
        return True
    if low.endswith(EXCLUDE_FILE_SUFFIXES):
        return True
    if "backup" in low or name.endswith("~"):
        return True
    return False


def _ignore(dirpath: str, names: list[str]) -> set[str]:
    """shutil.copytree ignore callback -- drops developer files/folders."""
    drop = set()
    for n in names:
        full = Path(dirpath) / n
        if full.is_dir() and n in EXCLUDE_DIRS:
            drop.add(n)
        elif full.is_file() and _excluded_file(n):
            drop.add(n)
    return drop


def build(platform: str) -> Path:
    name = TARGETS[platform]
    dest = OUT_ROOT / name
    print(f"\n=== building {name} ===")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for d in RUNTIME_DIRS:
        src = REPO / d
        if not src.exists():
            raise SystemExit(f"ERROR: missing runtime folder {src}")
        shutil.copytree(src, dest / d, ignore=_ignore)
        print(f"  copied {d}/")

    (dest / "docs").mkdir()
    for pdf in DOC_PDFS:
        src = REPO / "docs" / pdf
        if not src.exists():
            raise SystemExit(f"ERROR: missing PDF {src}")
        shutil.copy2(src, dest / "docs" / pdf)
        print(f"  copied docs/{pdf}")

    for f in ROOT_FILES:
        src = REPO / f
        if not src.exists():
            raise SystemExit(f"ERROR: missing root file {src}")
        shutil.copy2(src, dest / f)
        print(f"  copied {f}")

    pdir = REPO / "packaging" / platform
    for f in PLATFORM_FILES[platform]:
        src = pdir / f
        if not src.exists():
            raise SystemExit(f"ERROR: missing packaging file {src}")
        shutil.copy2(src, dest / f)
        print(f"  copied {f}")

    # Belt and braces: drop any __pycache__ that slipped through.
    for cache in dest.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    return dest


# ---------------------------------------------------------------- validation
FORBIDDEN_SUFFIXES = (".tex", ".aux", ".toc", ".log", ".out", ".pyc", ".pyo",
                      ".bak", ".orig", ".synctex.gz")
FORBIDDEN_DIRS = {"latex", "figures_src", "tests", ".git", ".worktrees",
                  "__pycache__", ".pytest_cache", "results", "build", "dist"}
FORBIDDEN_NAME_PARTS = ["first_user_audit", "backup", "packaging_review",
                        "_prompt"]
CODE_BAD_STRINGS = ["docs/latex", "docs\\latex", "figures_src", ".worktrees"]
REQUIRED = [
    "docs/ClassRoomSTORM_Help.pdf", "docs/ClassRoomSTORM_Theory.pdf",
    "apps/ClassRoomSTORM_Studio.py", "apps/ClassRoomSTORM_VirtualExperiment.py",
    "apps/ClassRoomSTORM_Reconstruction.py", "classroomstorm_core/__init__.py",
    "assets/experimental_led_matrix_video_2023-09-07.mp4",
    "requirements.txt", "requirements-gpu.txt", "LICENSE", "CITATION.cff",
    "README_FIRST.txt",
]


def validate(dest: Path) -> list[str]:
    problems: list[str] = []

    for p in dest.rglob("*"):
        rel = p.relative_to(dest).as_posix()
        if p.is_dir():
            if p.name in FORBIDDEN_DIRS:
                problems.append(f"forbidden directory: {rel}/")
        else:
            low = p.name.lower()
            if p.suffix.lower() in FORBIDDEN_SUFFIXES:
                problems.append(f"forbidden file type: {rel}")
            if any(part in low for part in FORBIDDEN_NAME_PARTS):
                problems.append(f"forbidden file name: {rel}")

    for r in REQUIRED:
        if not (dest / r).exists():
            problems.append(f"missing required file: {r}")

    for pdf in DOC_PDFS:
        f = dest / "docs" / pdf
        if f.exists() and f.read_bytes()[:5] != b"%PDF-":
            problems.append(f"not a valid PDF: docs/{pdf}")

    for p in (list(dest.rglob("*.py")) + list(dest.rglob("*.bat"))
              + list(dest.rglob("*.command"))):
        rel = p.relative_to(dest).as_posix()
        text = p.read_text(encoding="utf-8", errors="ignore")
        for bad in CODE_BAD_STRINGS:
            if bad in text:
                problems.append(f"code references developer path '{bad}': {rel}")

    # Syntax-check every shipped .py without writing any .pyc files.
    for p in dest.rglob("*.py"):
        rel = p.relative_to(dest).as_posix()
        try:
            compile(p.read_text(encoding="utf-8", errors="ignore"), str(p), "exec")
        except SyntaxError as exc:
            problems.append(f"syntax error in {rel}: {exc}")

    return problems


def write_manifest(dest: Path) -> int:
    rows = []
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.name != "PACKAGE_MANIFEST.txt":
            rel = p.relative_to(dest).as_posix()
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            rows.append(f"{digest}  {p.stat().st_size:>10d}  {rel}")
    (dest / "PACKAGE_MANIFEST.txt").write_text(
        "ClassRoomSTORM V1 user package manifest\n"
        f"files: {len(rows)}\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main(argv: list[str]) -> int:
    validate_only = "--validate" in argv
    all_ok = True
    for platform in TARGETS:
        dest = OUT_ROOT / TARGETS[platform]
        if not validate_only:
            dest = build(platform)
        if not dest.exists():
            print(f"  {TARGETS[platform]}: NOT FOUND")
            all_ok = False
            continue
        problems = validate(dest)
        if problems:
            all_ok = False
            print(f"  VALIDATION FAILED for {dest.name}:")
            for pr in problems:
                print(f"    - {pr}")
        else:
            count = write_manifest(dest) if not validate_only else \
                sum(1 for f in dest.rglob("*") if f.is_file())
            print(f"  {dest.name}: VALID ({count} files)")
    print()
    print("ALL PACKAGES OK" if all_ok else "PROBLEMS FOUND -- see above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
