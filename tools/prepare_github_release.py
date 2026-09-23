"""Prepare a source snapshot and user ZIPs without committing or publishing.

Only maintained software paths are exported. Local environments, results,
historical backups, Git metadata and build intermediates remain in place.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import stat
import zipfile

import build_user_packages as packages

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('.github', 'apps', 'assets', 'classroomstorm_core', 'docs',
               'examples', 'packaging', 'tests', 'tools')
SOURCE_FILES = ('.gitattributes', '.gitignore', 'README.md', 'CHANGELOG.md',
                'LICENSE', 'CITATION.cff', 'requirements.txt',
                'requirements-gpu.txt', 'requirements-dev.txt', 'requirements-figures.txt')
SKIP_DIRS = {'.git', '.venv', 'venv', '.worktrees', '__pycache__', '.pytest_cache',
             '.mypy_cache', '.ruff_cache', 'build', 'dist', 'results', 'qa', 'backups'}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.aux', '.log', '.out', '.toc', '.fls', '.fdb_latexmk',
                 '.bak', '.orig', '.tmp', '.pem', '.key'}
SKIP_NAMES = {'.DS_Store', 'Thumbs.db', 'desktop.ini', 'Scientific_Figure_Review.tex',
              'verification.json'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def exportable(relative):
    return (not any(part in SKIP_DIRS for part in relative.parts)
            and relative.suffix.lower() not in SKIP_SUFFIXES
            and relative.name not in SKIP_NAMES
            and not relative.name.startswith('.env')
            and not relative.name.endswith(('.synctex.gz', '~')))


def source_files(repo):
    for name in SOURCE_FILES:
        p = repo / name
        if not p.is_file():
            raise FileNotFoundError(f'Missing source file: {p}')
        yield p
    for folder in SOURCE_DIRS:
        if not (repo / folder).is_dir():
            raise FileNotFoundError(f'Missing source directory: {folder}')
        for p in sorted((repo / folder).rglob('*')):
            if exportable(p.relative_to(repo)):
                if p.is_symlink():
                    raise ValueError(f'Symlinks need explicit review before export: {p}')
                if p.is_file():
                    yield p


def export_source(repo, destination):
    paths = list(source_files(repo))
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for p in paths:
        if p.is_symlink():
            raise ValueError(f'Symlinks need explicit review before export: {p}')
        if p.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError(f'File exceeds the source-export size limit: {p}')
        relative = p.relative_to(repo)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        if target.suffix in ('.command', '.sh'):
            target.chmod(0o755)
        hashes[relative.as_posix()] = digest(target)
    manifest = {'version': packages.__version__, 'sha256': hashes}
    (destination / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    problems = validate_source(destination)
    if problems:
        raise ValueError('; '.join(problems))
    return manifest


def validate_source(destination):
    manifest = json.loads((destination / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(destination).as_posix(): digest(p)
              for p in destination.rglob('*') if p.is_file() and p.name != 'SOURCE_MANIFEST.json'}
    expected = manifest['sha256']
    return [f'Source manifest mismatch: {name}' for name in sorted(set(actual) | set(expected))
            if actual.get(name) != expected.get(name)]


def zip_tree(folder, target):
    """Use stable metadata and retain executable bits for macOS launchers."""
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(folder.rglob('*')):
            if not p.is_file():
                continue
            info = zipfile.ZipInfo((Path(folder.name) / p.relative_to(folder)).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            mode = 0o755 if p.suffix in ('.command', '.sh') else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, p.read_bytes())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'Archive integrity check failed: {target}')
    return digest(target)


def prepare(output):
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Choose an output directory outside the maintained source tree')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output must be new or empty; existing releases are preserved')
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'ClassRoomSTORM_Source'
    manifest = export_source(ROOT, source)
    original_root = packages.OUT_ROOT
    try:
        packages.OUT_ROOT = output
        for platform in packages.TARGETS:
            packages.build(platform)
    finally:
        packages.OUT_ROOT = original_root
    archives = {}
    for suffix, folder in [('source', source)] + [(platform, output / name) for platform, name in packages.TARGETS.items()]:
        target = output / f'ClassRoomSTORM-{packages.__version__}-{suffix}.zip'
        archives[target.name] = zip_tree(folder, target)
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{value}  {name}\n' for name, value in archives.items()), encoding='utf-8')
    changelog = (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    section = re.search(r'^## V' + re.escape(packages.__version__) + r'\b.*?(?=^## |\Z)', changelog, re.M | re.S)
    notes = f'# ClassRoomSTORM {packages.__version__}\n\nPrepared locally for a future GitHub update. No commit, push, tag or public release has been made by this tool.\n\n'
    notes += (section.group(0).strip() + '\n\n') if section else ''
    notes += ('Use `ClassRoomSTORM_Source/` as the repository content, keeping the destination repository\'s `.git` folder and history. '
              'Review changes on a branch before committing. Windows and macOS ZIPs are user downloads, not source folders. '
              'Follow `docs/RELEASE.md` in the source folder. `SHA256SUMS.txt` verifies the three ZIPs.\n')
    (output / 'RELEASE_NOTES.md').write_text(notes, encoding='utf-8')
    report = {'version': packages.__version__, 'source_files': len(manifest['sha256']),
              'archives': archives, 'published': False}
    (output / 'PREPARATION_MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path, help='New or empty directory for the prepared bundle')
    group.add_argument('--validate-source', type=Path, help='Verify an exported source tree without modifying it')
    args = parser.parse_args()
    if args.validate_source:
        problems = validate_source(args.validate_source)
        if problems:
            raise SystemExit('\n'.join(problems))
        print('Source manifest verified')
    else:
        print(json.dumps(prepare(args.output), indent=2))


if __name__ == '__main__':
    main()
