"""Exercise setup control flow without installing packages or using the network."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('offline', [True, False])
def test_setup_only_contacts_index_when_local_wheels_are_absent(tmp_path, offline):
    platform = 'windows' if sys.platform == 'win32' else 'macos'
    script = 'setup_windows.bat' if platform == 'windows' else 'setup_macos.command'
    shutil.copy2(ROOT / 'packaging' / platform / script, tmp_path / script)
    (tmp_path / 'requirements.txt').write_text('# deliberately empty\n')
    # There is no pip in this environment. Even a broken interception cannot
    # install a package. Python startup records pip arguments and exits first.
    venv.create(tmp_path / '.venv', with_pip=False)
    hook = tmp_path / 'hook'
    hook.mkdir()
    (hook / 'sitecustomize.py').write_text(
        "import json, os, sys\n"
        "if 'install' in sys.argv:\n"
        "    with open(os.environ['SETUP_TEST_LOG'], 'a') as f:\n"
        "        f.write(json.dumps(sys.argv) + '\\n')\n"
        "    os._exit(0)\n"
    )
    if offline:
        (tmp_path / 'wheels').mkdir()
        (tmp_path / 'wheels/fixture.whl').write_bytes(b'not installed')
    log = tmp_path / 'calls.jsonl'
    env = dict(os.environ, PYTHONPATH=str(hook), SETUP_TEST_LOG=str(log))
    command = ['cmd.exe', '/d', '/c', script] if platform == 'windows' else ['bash', script]
    run = subprocess.run(command, cwd=tmp_path, env=env, input='\n', text=True,
                         capture_output=True, timeout=40)
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'Setup completed' in run.stdout
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    if offline:
        assert len(calls) == 1
        assert '--no-index' in calls[0] and '--find-links' in calls[0]
        assert '--upgrade' not in calls[0]
    else:
        assert len(calls) == 2
        assert '--upgrade' in calls[0]
        assert '-r' in calls[1] and '--no-index' not in calls[1]
