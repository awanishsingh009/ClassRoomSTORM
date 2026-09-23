"""Protect private/local files and preserve usable, verifiable release snapshots."""
import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import prepare_github_release as release


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    for name in release.SOURCE_FILES:
        (repo / name).write_text('fixture\n', encoding='utf-8')
    for name in release.SOURCE_DIRS:
        (repo / name).mkdir()
    for name in ['apps/main.py', 'docs/figures_src/data/samples.csv',
                 'docs/figures_src/tikz/style.tex', 'docs/manual.pdf',
                 'packaging/macos/setup.command', 'tests/test_example.py']:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture\n', encoding='utf-8')
    return repo


def test_snapshot_keeps_editable_data_but_excludes_local_artifacts(source, tmp_path):
    junk = ['.git/config', '.venv/secret.txt', 'backups/old.py', 'qa/trace.json',
            'results/student.csv', 'docs/figures_src/build/old.pdf',
            'apps/__pycache__/main.pyc', 'apps/.env', 'docs/figures_src/verification.json']
    for name in junk:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('must stay local')
    dest = tmp_path / 'export'
    manifest = release.export_source(source, dest)
    assert release.validate_source(dest) == []
    assert 'docs/figures_src/data/samples.csv' in manifest['sha256']
    assert 'docs/figures_src/tikz/style.tex' in manifest['sha256']
    assert all(not (dest / name).exists() for name in junk)
    assert all((source / name).read_text() == 'must stay local' for name in junk)


def test_snapshot_manifest_detects_changes_and_unexpected_files(source, tmp_path):
    dest = tmp_path / 'export'
    release.export_source(source, dest)
    (dest / 'apps/main.py').write_text('changed')
    (dest / 'extra.txt').write_text('unexpected')
    (dest / 'LICENSE').unlink()
    failures = release.validate_source(dest)
    assert len(failures) == 3
    assert any('apps/main.py' in p for p in failures)
    assert any('extra.txt' in p for p in failures)
    assert any('LICENSE' in p for p in failures)


def test_archive_retains_mac_executable_bits_and_is_repeatable(source, tmp_path):
    first, second = tmp_path / 'first.zip', tmp_path / 'second.zip'
    assert release.zip_tree(source, first) == release.zip_tree(source, second)
    with zipfile.ZipFile(first) as archive:
        mode = archive.getinfo('repo/packaging/macos/setup.command').external_attr >> 16
        assert mode & 0o111 == 0o111
        assert archive.testzip() is None


def test_standalone_figure_paths_do_not_require_companion_workspace(source):
    folder = source / 'docs/figures_src'
    path = folder / 'project_paths.py'
    path.write_bytes((TOOLS.parent / 'docs/figures_src/project_paths.py').read_bytes())
    spec = importlib.util.spec_from_file_location('isolated_project_paths', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.HAS_MANUSCRIPT
    assert len(module.document_sources()) == 2
    assert module.available_figures([{'category':'theory'}, {'category':'manuscript'}]) == [{'category':'theory'}]
    assert not (source.parent / 'manuscript_writing').exists()
