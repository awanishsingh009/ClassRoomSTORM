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
import argparse
import datetime
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO.parent
sys.path.insert(0, str(REPO / "classroomstorm_core"))
from version import __version__

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


def _populate(platform: str, dest: Path) -> Path:
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

    return dest


def build(platform: str) -> Path:
    """Validate a staged package before replacing a destination; retain every old file."""
    root = OUT_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / TARGETS[platform]
    if dest.is_symlink() or dest.resolve().parent != root:
        raise ValueError("Package destination must be a direct, non-symlink child of the output directory")
    with tempfile.TemporaryDirectory(prefix=".package-stage-", dir=root) as stage_dir:
        staged = Path(stage_dir) / dest.name
        _populate(platform, staged)
        write_manifest(staged)
        problems = validate(staged, platform)
        if problems:
            raise ValueError("Staged package failed validation: " + "; ".join(problems))
        backup = None
        if dest.exists():
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive = root / "archive" / "user_packages" / stamp
            if not archive.resolve().is_relative_to(root):
                raise ValueError("Package archive must stay inside the output directory")
            archive.mkdir(parents=True)
            backup = archive / dest.name
            dest.rename(backup)
            print(f"  Previous package and all user results preserved in {backup}")
        try:
            staged.rename(dest)
        except OSError:
            if backup is not None:
                backup.rename(dest)
            raise
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


def validate(dest: Path, platform: str | None = None) -> list[str]:
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

    manifest_path = dest / "PACKAGE_MANIFEST.json"
    if not manifest_path.exists():
        problems.append("missing SHA-256 package manifest")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != __version__:
                problems.append("manifest version differs from the current source")
            recorded = manifest["sha256"]
            actual = {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in dest.rglob("*") if p.is_file() and not p.name.startswith("PACKAGE_MANIFEST.")}
            for rel in sorted(set(recorded) | set(actual)):
                if recorded.get(rel) != actual.get(rel):
                    problems.append(f"manifest mismatch: {rel}")
        except (ValueError, KeyError, TypeError):
            problems.append("invalid SHA-256 package manifest")
    if platform is not None:
        sources = {p.relative_to(REPO).as_posix(): p for folder in RUNTIME_DIRS
                   for p in (REPO / folder).rglob("*") if p.is_file()
                   and not any(part in EXCLUDE_DIRS for part in p.relative_to(REPO).parts)
                   and not _excluded_file(p.name)}
        sources.update({name: REPO / name for name in ROOT_FILES})
        sources.update({f"docs/{name}": REPO / "docs" / name for name in DOC_PDFS})
        sources.update({name: REPO / "packaging" / platform / name for name in PLATFORM_FILES[platform]})
        for rel, source in sources.items():
            target = dest / rel
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                problems.append(f"package differs from current source: {rel}")
    return problems


def write_manifest(dest: Path) -> int:
    rows = []
    hashes = {}
    for p in sorted(dest.rglob("*")):
        if p.is_file() and not p.name.startswith("PACKAGE_MANIFEST."):
            rel = p.relative_to(dest).as_posix()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            hashes[rel] = digest
            rows.append(f"{digest}  {p.stat().st_size:>10d}  {rel}")
    (dest / "PACKAGE_MANIFEST.txt").write_text(
        f"ClassRoomSTORM {__version__} user package manifest (SHA-256)\n"
        f"files: {len(rows)}\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    (dest / "PACKAGE_MANIFEST.json").write_text(json.dumps({"version": __version__, "sha256": hashes}, indent=2) + "\n", encoding="utf-8")
    return len(rows)


def main(argv: list[str]) -> int:
    global OUT_ROOT
    parser = argparse.ArgumentParser(description="Build validated user packages and preserve previous packages with their results.")
    parser.add_argument("--validate", action="store_true", help="Read-only manifest and source consistency check")
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--zip", action="store_true", help="Also create a versioned release ZIP after successful build")
    args = parser.parse_args(argv)
    OUT_ROOT = args.output_dir
    validate_only = args.validate
    all_ok = True
    for platform in TARGETS:
        dest = OUT_ROOT / TARGETS[platform]
        if not validate_only:
            dest = build(platform)
        if not dest.exists():
            print(f"  {TARGETS[platform]}: NOT FOUND")
            all_ok = False
            continue
        problems = validate(dest, platform)
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
    if all_ok and args.zip:
        release_dir = OUT_ROOT / "releases"
        release_dir.mkdir(parents=True, exist_ok=True)
        target = release_dir / f"ClassRoomSTORM_{__version__}.zip"
        if target.exists():
            raise FileExistsError(f"Release already exists; preserve it and choose another output directory: {target}")
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            for name in TARGETS.values():
                for path in sorted((OUT_ROOT / name).rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(OUT_ROOT))
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise ValueError("Release ZIP failed its integrity check")
        print(f"Release: {target}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
