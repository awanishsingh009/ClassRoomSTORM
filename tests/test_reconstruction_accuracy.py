"""Regression checks for coordinate, validation, and quantitative output contracts."""
import csv
import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "classroomstorm_core"))
from reconstruction import reconstruct_frames, render_localization_image, write_localizations_csv
from truth_comparison import match_localizations_to_truth, nearest_truth_errors


def test_application_style_import_and_vector_diagram_fallback(tmp_path, monkeypatch):
    # Workflow apps import modules directly from the core directory, whereas
    # numerical documentation imports classroomstorm_core as a package.
    import plotting
    import pipeline_report
    monkeypatch.delenv("CLASSROOMSTORM_DISABLE_MATPLOTLIB", raising=False)
    assert plotting._try_matplotlib() is not None
    path = tmp_path / "equations.png"
    pipeline_report._save_equation_map(path, {
        "threshold_quantile": .995, "background_mode": "frame_median",
        "blinker_mode_resolved": "multiple", "processing_backend_used": "serial",
    })
    assert path.stat().st_size > 15000
    assert "<svg" in path.with_suffix(".svg").read_text(encoding="utf-8")


def spot(x=10, y=12, background=0):
    frame = np.full((24, 24), background, dtype=float)
    frame[y, x] += 100
    return frame


def test_crop_preserves_camera_coordinates_and_applies_calibration(tmp_path):
    result = reconstruct_frames([spot()], crop=(5, 7, 14, 14), pixel_size=0.25, unit="mm")
    loc = result.localizations[0]
    assert (loc["x_px"], loc["y_px"]) == (5, 5)
    assert (loc["x_full_px"], loc["y_full_px"]) == (10, 12)
    assert (loc["x_calibrated"], loc["y_calibrated"], loc["unit"]) == (2.5, 3, "mm")
    truth = [{"frame": 0, "x_px": 10, "y_px": 12}]
    assert nearest_truth_errors(result.localizations, truth) == [0]
    path = tmp_path / "localizations.csv"
    write_localizations_csv(path, result.localizations)
    with path.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["x_calibrated"]) == 2.5


def test_negative_crop_is_intersection_not_translation():
    result = reconstruct_frames([spot(2, 3)], crop=(-5, -7, 15, 17))
    assert result.raw_mean.shape == (10, 10)
    assert result.summary["crop"] == {"x": 0, "y": 0, "width": 10, "height": 10}


@pytest.mark.parametrize("mode", ["single", "multiple", "auto"])
@pytest.mark.parametrize("level", [0, 17])
def test_blank_frames_do_not_invent_emitters(mode, level):
    result = reconstruct_frames([np.full((12, 12), level)], blinker_mode=mode)
    assert result.localizations == []
    assert result.summary["frames_without_localizations"] == 1
    assert not result.superres.any()


@pytest.mark.parametrize("quantile", [0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_quantiles_are_rejected(quantile):
    with pytest.raises(ValueError, match="threshold_quantile"):
        reconstruct_frames([spot()], threshold_quantile=quantile)


@pytest.mark.parametrize("frame", [np.zeros((0, 4)), np.zeros((3, 3, 3)), np.full((4, 4), np.nan), np.full((4, 4), np.inf)])
def test_malformed_frames_fail_with_frame_index(frame):
    with pytest.raises(ValueError, match="frame 0"):
        reconstruct_frames([frame])


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_invalid_calibration_is_rejected(scale):
    with pytest.raises(ValueError, match="pixel_size"):
        reconstruct_frames([spot()], pixel_size=scale)


def test_background_correction_preserves_raw_mean():
    clean = reconstruct_frames([spot()], threshold_quantile=1)
    corrected = reconstruct_frames([spot(background=25)], threshold_quantile=1, background_mode="frame_median")
    assert corrected.localizations == clean.localizations
    np.testing.assert_allclose(corrected.raw_mean, clean.raw_mean + 25)
    assert corrected.summary["mean_subtracted_background"] == 25


def test_render_preserves_fractional_positions_and_unit_mass():
    a = render_localization_image([{"x_px": 10.1, "y_px": 12.1}], (30, 30))
    b = render_localization_image([{"x_px": 10.4, "y_px": 12.4}], (30, 30))
    assert not np.array_equal(a, b)
    yy, xx = np.mgrid[:30, :30]
    assert (b * xx).sum() == pytest.approx(10.4, abs=1e-4)
    assert (b * yy).sum() == pytest.approx(12.4, abs=1e-4)
    assert a.sum() == pytest.approx(1)
    edge = render_localization_image([{"x_px": 0.1, "y_px": 0.1}], (30, 30))
    assert edge.sum() == pytest.approx(1)


def test_matching_counts_false_positives_misses_and_frame_identity():
    locs = [{"frame": 0, "x_px": 1, "y_px": 1},
            {"frame": 0, "x_px": 1.1, "y_px": 1},
            {"frame": 1, "x_px": 8, "y_px": 8}]
    truth = [{"frame": 0, "x_px": 1, "y_px": 1},
             {"frame": 0, "x_px": 8, "y_px": 8}]
    stats = match_localizations_to_truth(locs, truth, frames_processed=2)
    assert stats["matched_localizations"] == 1
    assert stats["false_positives"] == 2
    assert stats["false_negatives"] == 1
    assert stats["precision"] == pytest.approx(1 / 3)
    assert stats["recall"] == 0.5


def test_matching_finds_maximum_cardinality_not_greedy_count():
    locs = [{"x_px": 0.1, "y_px": 0}, {"x_px": -0.2, "y_px": 0}]
    truth = [{"x_px": 0, "y_px": 0}, {"x_px": 0.9, "y_px": 0}]
    assert match_localizations_to_truth(locs, truth, 0.85)["matched_localizations"] == 2


def test_truth_scope_excludes_unprocessed_frames_and_outside_crop():
    truth = [{"frame": 0, "x_px": 10, "y_px": 12},
             {"frame": 0, "x_px": 1, "y_px": 1},
             {"frame": 2, "x_px": 10, "y_px": 12}]
    result = reconstruct_frames([spot()], crop=(5, 7, 14, 14))
    stats = match_localizations_to_truth(result.localizations, truth, frames_processed=1, crop=result.summary["crop"])
    assert stats["evaluated_truth_events"] == 1
    assert stats["precision"] == stats["recall"] == stats["f1"] == 1


def test_empty_matching_is_json_safe():
    stats = match_localizations_to_truth([], [], frames_processed=1)
    assert stats["precision"] is None
    json.dumps(stats, allow_nan=False)


def app_module():
    spec = importlib.util.spec_from_file_location("reconstruction_accuracy_app", ROOT / "apps/ClassRoomSTORM_Reconstruction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_result_directory_is_protected(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "results"
    out.mkdir()
    original = out / "summary.json"
    original.write_text('{"original": true}')
    with pytest.raises(ValueError, match="not empty"):
        app_module().run_reconstruction(video, out)
    assert original.read_text() == '{"original": true}'


def test_app_exports_crop_aware_metrics_and_calibration(tmp_path):
    app = app_module()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video fixture")
    (tmp_path / "truth.csv").write_text("frame,x_px,y_px,active\n0,10,12,1\n")
    out = tmp_path / "out"
    plot_names = ["save_grayscale_png", "save_superres_png", "save_side_by_side", "save_localization_scatter", "save_density_map", "save_raw_localization_overlay", "save_cumulative_reconstruction_panel", "generate_pipeline_report"]
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(app, "read_video_frames", return_value=[spot()]))
        for name in plot_names:
            stack.enter_context(mock.patch.object(app, name))
        summary = app.run_reconstruction(video, out, crop=(5, 7, 14, 14), compare_truth=True, pixel_size=0.5, unit="mm")
    assert summary["truth_comparison"]["f1"] == 1
    assert summary["truth_comparison"]["mean_matched_error_px"] == 0
    assert summary["calibration"]["applied"]
    assert len(summary["input_sha256"]) == 64
    assert json.loads((out / "summary.json").read_text())["truth_used_for_reconstruction"] is False


@pytest.mark.parametrize("kwargs", [{"frames": 0}, {"sigma_px": 0}, {"fps": float("nan")}, {"background": -1}])
def test_virtual_generator_rejects_invalid_parameters_before_writing(tmp_path, kwargs):
    spec = importlib.util.spec_from_file_location("virtual_accuracy_app", ROOT / "apps/ClassRoomSTORM_VirtualExperiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "new"
    with pytest.raises(ValueError):
        module.generate_dataset(out, **kwargs)
    assert not out.exists()


def test_package_imports_work_without_legacy_sys_path(tmp_path):
    import subprocess
    code = ("import sys; sys.path.insert(0, " + repr(str(ROOT)) + "); "
            "import numpy as np; from classroomstorm_core.plotting import save_grayscale_png; "
            "save_grayscale_png('image.png', np.eye(3))")
    run = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "image.png").exists()
