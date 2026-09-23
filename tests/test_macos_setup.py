"""Exercise Mac setup with old system Python, failed environments and spaces in paths."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import venv

import pytest

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='Runs native POSIX shell scripts')
ROOT = Path(__file__).resolve().parents[1]


def python_wrapper(path, version):
    """Execute the real version check with a simulated interpreter version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = f'import sys; sys.version_info = {version!r}; exec(sys.argv[1])'
    path.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = "-c" ]; then\n'
        f'  exec {shlex.quote(sys.executable)} -c {shlex.quote(probe)} "$2"\n'
        'fi\n'
        'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then\n'
        f'  exec {shlex.quote(sys.executable)} -m venv --without-pip "$3"\n'
        'fi\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n', encoding='utf-8')
    path.chmod(0o755)
    return path


@pytest.fixture
def package(tmp_path):
    folder = tmp_path / 'student package'
    folder.mkdir()
    for name in ('setup_macos.command', 'launch_studio_macos.command'):
        shutil.copy2(ROOT / 'packaging/macos' / name, folder / name)
    (folder / 'requirements.txt').write_text('# no downloads in this test\n')
    bin_dir = tmp_path / 'test bin'
    python_wrapper(bin_dir / 'python3', (3, 9, 6))
    python_wrapper(bin_dir / 'python3.11', (3, 11, 0))
    hook = tmp_path / 'hook'
    hook.mkdir()
    (hook / 'sitecustomize.py').write_text(
        'import json, os, sys\n'
        "if 'install' in sys.argv:\n"
        "    with open(os.environ['SETUP_TEST_LOG'], 'a') as f:\n"
        "        f.write(json.dumps(sys.argv) + '\\n')\n"
        '    os._exit(0)\n')
    log = tmp_path / 'pip-calls.jsonl'
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
               PYTHONPATH=str(hook), SETUP_TEST_LOG=str(log))
    env.pop('CLASSROOMSTORM_PYTHON', None)
    return folder, env, log


def run_script(package, script='setup_macos.command'):
    folder, env, _ = package
    return subprocess.run(['/bin/bash', str(folder / script)], cwd=folder.parent,
                          env=env, input='\n', text=True, capture_output=True, timeout=40)


@pytest.mark.parametrize('version', [(3, 9, 6), (3, 14, 0)])
def test_unsupported_python_stops_before_modifying_old_environment(package, version):
    folder, env, log = package
    python_wrapper(folder / '.venv/bin/python', version)
    marker = folder / '.venv/keep.txt'
    marker.write_text('preserve me')
    env['CLASSROOMSTORM_PYTHON'] = str(folder / '.venv/bin/python')
    result = run_script(package)
    assert result.returncode == 1
    assert 'No supported Python was found' in result.stdout
    assert 'Setup completed' not in result.stdout
    assert marker.read_text() == 'preserve me'
    assert not list(folder.glob('.venv-backup-*'))
    assert not log.exists()


def test_selects_versioned_python_and_preserves_old_environment(package):
    folder, _, log = package
    python_wrapper(folder / '.venv/bin/python', (3, 9, 6))
    (folder / '.venv/keep.txt').write_text('old environment')
    (folder / 'results').mkdir()
    (folder / 'results/student.csv').write_text('student work')
    result = run_script(package)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Using Python: python3.11' in result.stdout
    assert 'Setup completed' in result.stdout
    backups = list(folder.glob('.venv-backup-*'))
    assert len(backups) == 1
    assert (backups[0] / 'keep.txt').read_text() == 'old environment'
    assert (folder / 'results/student.csv').read_text() == 'student work'
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    assert '--no-compile' in calls[-1]


def test_valid_existing_environment_is_reused(package):
    folder, env, log = package
    venv.create(folder / '.venv', with_pip=False)
    (folder / '.venv/keep.txt').write_text('valid environment')
    env['CLASSROOMSTORM_PYTHON'] = '/nonexistent/python'
    result = run_script(package)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (folder / '.venv/keep.txt').read_text() == 'valid environment'
    assert not list(folder.glob('.venv-backup-*'))
    assert log.exists()


@pytest.mark.parametrize('version', [None, (3, 9, 6), (3, 14, 0)])
def test_launcher_requests_setup_for_missing_or_unsupported_environment(package, version):
    folder, _, _ = package
    if version is not None:
        python_wrapper(folder / '.venv/bin/python', version)
    result = run_script(package, 'launch_studio_macos.command')
    assert result.returncode == 1
    assert 'setup_macos.command' in result.stdout
    assert 'ClassRoomSTORM_Studio.py' not in result.stderr
