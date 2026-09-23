"""Packaging must preserve existing user data and detect stale or damaged files."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def builder(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools/build_user_packages.py"
    spec = importlib.util.spec_from_file_location("test_package_builder", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO = tmp_path / "source"
    module.OUT_ROOT = tmp_path / "output"
    for folder in module.RUNTIME_DIRS:
        (module.REPO / folder).mkdir(parents=True)
    for rel in module.REQUIRED + module.ROOT_FILES:
        if rel == "README_FIRST.txt":
            continue
        p = module.REPO / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"%PDF-fixture" if p.suffix == ".pdf" else b"# fixture\n")
    for platform, names in module.PLATFORM_FILES.items():
        for name in names:
            p = module.REPO / "packaging" / platform / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# fixture\n")
    return module


def test_rebuild_preserves_previous_results(builder):
    old = builder.OUT_ROOT / builder.TARGETS["windows"]
    (old / "results").mkdir(parents=True)
    (old / "results/student.csv").write_text("irreplaceable result")
    new = builder.build("windows")
    assert builder.validate(new, "windows") == []
    preserved = list((builder.OUT_ROOT / "archive").rglob("student.csv"))
    assert len(preserved) == 1
    assert preserved[0].read_text() == "irreplaceable result"
    assert not (new / "results").exists()


def test_manifest_and_source_checks_detect_tampering(builder):
    dest = builder.build("macos")
    (dest / "apps/ClassRoomSTORM_Studio.py").write_text("# changed\n")
    problems = builder.validate(dest, "macos")
    assert any("manifest mismatch" in p for p in problems)
    assert any("differs from current source" in p for p in problems)


def test_failed_staging_leaves_previous_package_untouched(builder):
    dest = builder.OUT_ROOT / builder.TARGETS["windows"]
    dest.mkdir(parents=True)
    (dest / "keep.txt").write_text("keep")
    (builder.REPO / "docs/ClassRoomSTORM_Help.pdf").write_text("not a PDF")
    with pytest.raises(ValueError, match="validation"):
        builder.build("windows")
    assert (dest / "keep.txt").read_text() == "keep"
