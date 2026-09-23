import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import sys
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "classroomstorm_core"
sys.path.insert(0, str(SHARED))
RECON_APP = ROOT / "apps" / "ClassRoomSTORM_Reconstruction.py"
VIRTUAL_APP = ROOT / "apps" / "ClassRoomSTORM_VirtualExperiment.py"
STUDIO_APP = ROOT / "apps" / "ClassRoomSTORM_Studio.py"


def load_reconstruction_app():
    spec = importlib.util.spec_from_file_location("ClassRoomSTORM_Reconstruction", RECON_APP)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_virtual_app():
    spec = importlib.util.spec_from_file_location("ClassRoomSTORM_VirtualExperiment", VIRTUAL_APP)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_studio_app():
    spec = importlib.util.spec_from_file_location("ClassRoomSTORM_Studio", STUDIO_APP)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ClassRoomSTORMCoreTests(unittest.TestCase):
    def test_led_text_renders_each_character_as_16_by_16_cell(self):
        from led_text import CELL_SIZE, text_to_led_mask

        mask = text_to_led_mask("ABBE")

        self.assertEqual(CELL_SIZE, 16)
        self.assertEqual(mask.shape, (16, 64))
        self.assertGreater(int(mask.sum()), 0)
        self.assertGreater(int(mask[:, 0:16].sum()), 0)
        self.assertGreater(int(mask[:, 16:32].sum()), 0)
        self.assertGreater(int(mask[:, 32:48].sum()), 0)
        self.assertGreater(int(mask[:, 48:64].sum()), 0)

    def test_led_text_space_keeps_blank_16_by_16_cell(self):
        from led_text import text_to_led_mask

        mask = text_to_led_mask("A E")

        self.assertEqual(mask.shape, (16, 48))
        self.assertEqual(int(mask[:, 16:32].sum()), 0)

    def test_virtual_text_mode_uses_fixed_16_by_16_led_alphabet(self):
        app_module = load_virtual_app()

        mask = app_module.build_source_mask("text", "ABBE", "", "single")

        self.assertEqual(mask.shape, (16, 64))
        self.assertEqual(app_module.TEXT_MASK_MODE, "fixed_16x16_led_alphabet")

    def test_logo_mask_scaling_changes_logo_resolution_without_changing_text(self):
        app_module = load_virtual_app()

        mask = np.ones((10, 20), dtype=np.uint8)
        scaled_up = app_module.scale_binary_mask(mask, 2.0)
        scaled_down = app_module.scale_binary_mask(mask, 0.5)
        text_mask = app_module.build_source_mask("text", "ABBE", "", "single")

        self.assertEqual(scaled_up.shape, (20, 40))
        self.assertEqual(scaled_down.shape, (5, 10))
        self.assertEqual(text_mask.shape, (16, 64))

    def test_logo_grid_sampling_distributes_source_points(self):
        app_module = load_virtual_app()

        mask = np.ones((12, 12), dtype=np.uint8)
        sampled = app_module.sample_logo_source_mask(mask, method="grid", grid_step_px=4, max_source_points=0, seed=1)

        self.assertLess(int(sampled.sum()), int(mask.sum()))
        self.assertEqual(int(sampled.sum()), 9)
        active_y, active_x = np.where(sampled > 0)
        self.assertEqual(len(set(active_y // 4)), 3)
        self.assertEqual(len(set(active_x // 4)), 3)

    def test_virtual_text_defaults_are_strong_for_sigma_10(self):
        app_module = load_virtual_app()

        defaults = app_module.default_virtual_settings()

        self.assertEqual(defaults["sigma_px"], 10.0)
        self.assertEqual(defaults["frames"], 1000)
        self.assertEqual(defaults["photons"], 80000.0)
        self.assertEqual(defaults["background"], 1.0)
        self.assertEqual(defaults["read_noise"], 0.2)
        self.assertEqual(defaults["blinkers_per_frame"], 30)
        self.assertEqual(defaults["logo_scale"], 1.0)
        self.assertEqual(defaults["logo_sampling"], "grid")
        self.assertEqual(defaults["logo_grid_step_px"], 1)

    def test_blinker_spacing_mode_converts_sigma_to_minimum_distance(self):
        app_module = load_virtual_app()

        self.assertEqual(app_module.blinker_min_distance_px("random", 10.0, manual_distance_px=100.0), 0.0)
        self.assertEqual(app_module.blinker_min_distance_px("overlapping", 10.0, manual_distance_px=100.0), 75.0)
        self.assertEqual(app_module.blinker_min_distance_px("touching", 10.0, manual_distance_px=100.0), 100.0)
        self.assertEqual(app_module.blinker_min_distance_px("separated", 10.0, manual_distance_px=100.0), 150.0)
        self.assertEqual(app_module.blinker_min_distance_px("separated", 10.0, manual_distance_px=0.0), 30.0)

    def test_multiple_blinker_brightness_scales_from_requested_count(self):
        app_module = load_virtual_app()

        self.assertEqual(app_module.effective_photons_per_blinker(80000.0, 1), 80000.0)
        self.assertEqual(app_module.effective_photons_per_blinker(80000.0, 4), 80000.0)
        self.assertEqual(app_module.effective_photons_per_blinker(80000.0, 30), 600000.0)

    def test_parallel_virtual_generation_matches_serial_truth(self):
        app_module = load_virtual_app()

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            serial_meta = app_module.generate_dataset(
                base / "serial",
                pattern="dots",
                dot_mode="pair",
                frames=6,
                blinkers_per_frame=2,
                sigma_px=2.0,
                photons=1000.0,
                background=1.0,
                read_noise=0.0,
                worker_count=1,
                seed=44,
            )
            parallel_meta = app_module.generate_dataset(
                base / "parallel",
                pattern="dots",
                dot_mode="pair",
                frames=6,
                blinkers_per_frame=2,
                sigma_px=2.0,
                photons=1000.0,
                background=1.0,
                read_noise=0.0,
                worker_count=2,
                seed=44,
            )

            self.assertEqual((base / "serial" / "truth.csv").read_text(encoding="utf-8"), (base / "parallel" / "truth.csv").read_text(encoding="utf-8"))
            self.assertEqual(serial_meta["frame_generation_backend_used"], "serial")
            self.assertEqual(parallel_meta["frame_generation_backend_used"], "cpu_parallel")
            self.assertEqual(parallel_meta["frame_generation_worker_count"], 2)
            self.assertEqual(parallel_meta["actual_mean_blinkers_per_frame"], serial_meta["actual_mean_blinkers_per_frame"])

    def test_auto_cpu_workers_use_at_most_ninety_percent_of_cpu_count(self):
        app_module = load_virtual_app()
        from reconstruction import _default_worker_count

        with mock.patch("os.cpu_count", return_value=16):
            self.assertEqual(app_module.virtual_worker_count(0, 100), 14)
            self.assertEqual(_default_worker_count(), 14)

    def test_virtual_frame_backend_resolves_gpu_or_cpu_fallback(self):
        app_module = load_virtual_app()

        with mock.patch.object(app_module, "_torch_cuda_status", return_value=(True, "Test CUDA GPU")):
            used, workers, note, device = app_module.resolve_virtual_frame_backend("gpu_cuda", 0, 25)
            self.assertEqual(used, "gpu_cuda")
            self.assertEqual(workers, 1)
            self.assertIsNone(note)
            self.assertEqual(device, "Test CUDA GPU")

        with mock.patch.object(app_module, "_torch_cuda_status", return_value=(False, "CUDA unavailable")), mock.patch("os.cpu_count", return_value=16):
            used, workers, note, device = app_module.resolve_virtual_frame_backend("gpu_cuda", 0, 25)
            self.assertEqual(used, "cpu_parallel")
            self.assertEqual(workers, 14)
            self.assertIn("CUDA unavailable", note)
            self.assertIsNone(device)

    def test_spaced_blinker_selection_treats_count_as_maximum(self):
        from simulation import BlinkerPoint, choose_active_points

        points = [BlinkerPoint(x_px=float(x), y_px=0.0, emitter_id=i) for i, x in enumerate(range(0, 100, 10))]
        rng = np.random.default_rng(4)

        active = choose_active_points(
            points,
            rng,
            mode="multiple",
            blinkers_per_frame=10,
            spacing_mode="separated",
            min_distance_px=25.0,
        )

        self.assertLess(len(active), 10)
        for i, left in enumerate(active):
            for right in active[i + 1:]:
                distance = ((left.x_px - right.x_px) ** 2 + (left.y_px - right.y_px) ** 2) ** 0.5
                self.assertGreaterEqual(distance, 25.0)

    def test_adaptive_padding_scales_with_sigma(self):
        from simulation import compute_edge_padding_px

        self.assertEqual(compute_edge_padding_px(1.0), 20)
        self.assertEqual(compute_edge_padding_px(4.0), 20)
        self.assertEqual(compute_edge_padding_px(8.2), 41)

    def test_source_points_are_written_in_padded_camera_coordinates(self):
        from simulation import mask_to_camera_points

        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        points, canvas_shape = mask_to_camera_points(mask, spacing_px=10.0, padding_px=25)

        self.assertEqual(canvas_shape, (61, 61))
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0].x_px, 25.0)
        self.assertAlmostEqual(points[0].y_px, 25.0)
        self.assertAlmostEqual(points[1].x_px, 35.0)
        self.assertAlmostEqual(points[1].y_px, 35.0)

    def test_rendered_edge_spot_has_blank_padding_space(self):
        from simulation import compute_edge_padding_px, mask_to_camera_points, render_gaussian_frame

        sigma_px = 7.0
        padding_px = compute_edge_padding_px(sigma_px)
        mask = np.array([[1]], dtype=np.uint8)
        points, canvas_shape = mask_to_camera_points(mask, spacing_px=12.0, padding_px=padding_px)
        frame = render_gaussian_frame(points, canvas_shape, sigma_px=sigma_px, photons=1000, background=0, read_noise=0, seed=1)

        self.assertEqual(frame.shape, canvas_shape)
        self.assertGreater(points[0].x_px, 4 * sigma_px)
        self.assertGreater(points[0].y_px, 4 * sigma_px)
        self.assertGreater(float(frame.sum()), 900.0)

    def test_metadata_round_trip(self):
        from metadata import write_metadata, read_metadata

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            data = {
                "generator_name": "ClassRoomSTORM Virtual Experiment",
                "generator_version": "V1",
                "frames": 12,
                "sigma_px": 3.5,
                "edge_padding_px": 20,
            }
            write_metadata(path, data)
            loaded = read_metadata(path)

        self.assertEqual(loaded["generator_version"], "V1")
        self.assertEqual(loaded["frames"], 12)
        self.assertEqual(loaded["sigma_px"], 3.5)

    def test_simple_reconstruction_from_synthetic_frames(self):
        from reconstruction import reconstruct_frames
        from simulation import BlinkerPoint, render_gaussian_frame

        frames = []
        truth = [(30.0, 25.0), (45.0, 40.0), (60.0, 55.0)]
        for i, (x, y) in enumerate(truth):
            frame = render_gaussian_frame(
                [BlinkerPoint(x_px=x, y_px=y, emitter_id=i)],
                (90, 90),
                sigma_px=2.2,
                photons=4000,
                background=2,
                read_noise=0,
                seed=10 + i,
            )
            frames.append(frame)

        result = reconstruct_frames(frames, threshold_quantile=0.995, multi_emitter=False)

        self.assertEqual(len(result.localizations), 3)
        for loc, (x_true, y_true) in zip(result.localizations, truth):
            self.assertLess(abs(loc["x_px"] - x_true), 1.0)
            self.assertLess(abs(loc["y_px"] - y_true), 1.0)
        self.assertEqual(result.raw_mean.shape, (90, 90))
        self.assertEqual(result.superres.shape, (90, 90))

    def test_parallel_reconstruction_matches_serial_localization_count(self):
        from reconstruction import reconstruct_frames

        frames = []
        for y, x in [(10, 12), (11, 13), (10, 14), (12, 15), (13, 16), (14, 17)]:
            frame = np.zeros((24, 24), dtype=float)
            frame[y, x] = 1000.0
            frames.append(frame)

        serial = reconstruct_frames(frames, threshold_quantile=0.99, blinker_mode="single", processing_backend="serial")
        parallel = reconstruct_frames(frames, threshold_quantile=0.99, blinker_mode="single", processing_backend="cpu_parallel", worker_count=2)

        self.assertEqual(len(parallel.localizations), len(serial.localizations))
        self.assertEqual(parallel.summary["processing_backend_requested"], "cpu_parallel")
        self.assertEqual(parallel.summary["processing_backend_used"], "cpu_parallel")
        self.assertEqual(parallel.summary["worker_count"], 2)

    def test_gpu_experimental_uses_gpu_when_available_or_records_fallback(self):
        from reconstruction import reconstruct_frames

        frames = []
        for y, x in [(8, 8), (9, 9), (10, 10)]:
            frame = np.zeros((18, 18), dtype=float)
            frame[y, x] = 500.0
            frames.append(frame)

        result = reconstruct_frames(frames, threshold_quantile=0.99, blinker_mode="single", processing_backend="gpu_experimental", worker_count=1)

        self.assertEqual(result.summary["processing_backend_requested"], "gpu_experimental")
        self.assertIn(result.summary["processing_backend_used"], {"gpu_torch", "serial", "cpu_parallel"})
        if result.summary["processing_backend_used"] == "gpu_torch":
            self.assertIn("gpu_device", result.summary)
        else:
            self.assertIn("gpu_note", result.summary)

    def test_gpu_experimental_multiple_mode_records_cpu_fallback(self):
        from reconstruction import reconstruct_frames

        frame = np.zeros((18, 18), dtype=float)
        frame[5, 5] = 500.0
        frame[12, 12] = 500.0

        result = reconstruct_frames([frame], threshold_quantile=0.99, blinker_mode="multiple", processing_backend="gpu_experimental", worker_count=1)

        self.assertEqual(result.summary["processing_backend_requested"], "gpu_experimental")
        self.assertEqual(result.summary["processing_backend_used"], "serial")
        self.assertIn("single-emitter", result.summary["gpu_note"])

    def test_hybrid_cpu_gpu_uses_gpu_for_single_or_cpu_for_multiple(self):
        from reconstruction import reconstruct_frames

        single_frames = []
        for y, x in [(8, 8), (9, 9), (10, 10)]:
            frame = np.zeros((18, 18), dtype=float)
            frame[y, x] = 500.0
            single_frames.append(frame)
        single = reconstruct_frames(single_frames, threshold_quantile=0.99, blinker_mode="single", processing_backend="hybrid_cpu_gpu", worker_count=2)

        self.assertEqual(single.summary["processing_backend_requested"], "hybrid_cpu_gpu")
        self.assertIn(single.summary["processing_backend_used"], {"gpu_torch", "cpu_parallel", "serial"})

        multi_frame = np.zeros((18, 18), dtype=float)
        multi_frame[5, 5] = 500.0
        multi_frame[12, 12] = 500.0
        multiple = reconstruct_frames([multi_frame, multi_frame], threshold_quantile=0.99, blinker_mode="multiple", processing_backend="hybrid_cpu_gpu", worker_count=2)

        self.assertEqual(multiple.summary["processing_backend_requested"], "hybrid_cpu_gpu")
        self.assertEqual(multiple.summary["processing_backend_used"], "cpu_parallel")
        self.assertIn("Hybrid CPU+GPU", multiple.summary["gpu_note"])

    def test_auto_cpu_gpu_selects_available_acceleration_after_mode_resolution(self):
        from reconstruction import reconstruct_frames

        single_frames = []
        for y, x in [(8, 8), (9, 9), (10, 10)]:
            frame = np.zeros((18, 18), dtype=float)
            frame[y, x] = 500.0
            single_frames.append(frame)
        single = reconstruct_frames(single_frames, threshold_quantile=0.99, blinker_mode="single", processing_backend="auto_cpu_gpu", worker_count=2)

        self.assertEqual(single.summary["processing_backend_requested"], "auto_cpu_gpu")
        self.assertIn(single.summary["processing_backend_used"], {"gpu_torch", "cpu_parallel", "serial"})

        multi_frame = np.zeros((18, 18), dtype=float)
        multi_frame[5, 5] = 500.0
        multi_frame[12, 12] = 500.0
        multiple = reconstruct_frames([multi_frame, multi_frame], threshold_quantile=0.99, blinker_mode="multiple", processing_backend="auto_cpu_gpu", worker_count=2)

        self.assertEqual(multiple.summary["processing_backend_requested"], "auto_cpu_gpu")
        self.assertEqual(multiple.summary["processing_backend_used"], "cpu_parallel")
        self.assertIn("Auto CPU+GPU", multiple.summary["gpu_note"])

    def test_auto_cpu_gpu_acceleration_uses_gpu_thresholding_for_multiple_mode(self):
        from reconstruction import reconstruct_frames

        frames = []
        for shift in range(3):
            frame = np.zeros((18, 18), dtype=float)
            frame[5, 5 + shift] = 500.0
            frame[12, 12] = 600.0
            frames.append(frame)

        result = reconstruct_frames(frames, threshold_quantile=0.99, blinker_mode="multiple", processing_backend="auto_cpu_gpu_acceleration", worker_count=2)

        self.assertEqual(result.summary["processing_backend_requested"], "auto_cpu_gpu_acceleration")
        self.assertIn(result.summary["processing_backend_used"], {"gpu_threshold_cpu_components", "cpu_parallel", "serial"})
        self.assertIn("timing_seconds", result.summary)
        if result.summary["processing_backend_used"] == "gpu_threshold_cpu_components":
            self.assertIn("gpu_device", result.summary)

    def test_reconstruction_app_exposes_processing_backend_options(self):
        app_module = load_reconstruction_app()

        self.assertEqual(
            app_module.processing_backend_options(),
            ["serial", "auto_cpu", "gpu_only", "auto_cpu_gpu", "auto_cpu_gpu_acceleration"],
        )

    def test_reconstruction_app_exposes_gpu_dependency_guidance(self):
        app_module = load_reconstruction_app()

        cpu_text = app_module.gpu_dependency_guidance("serial")
        gpu_text = app_module.gpu_dependency_guidance("gpu_only")

        self.assertIn("No GPU dependency", cpu_text)
        self.assertIn("NVIDIA", gpu_text)
        self.assertIn("requirements-gpu.txt", gpu_text)
        self.assertIn("AMD", gpu_text)
        self.assertIn("Intel", gpu_text)
        self.assertIn("MacBook", gpu_text)
        self.assertIn("not currently implemented", gpu_text)

    def test_reconstruction_preview_zoom_helpers_are_bounded_and_labeled(self):
        app_module = load_reconstruction_app()

        self.assertEqual(app_module.clamp_preview_zoom(0.05), 0.25)
        self.assertEqual(app_module.clamp_preview_zoom(10.0), 4.0)
        self.assertEqual(app_module.clamp_preview_zoom(1.249), 1.25)
        self.assertEqual(app_module.preview_zoom_label(1.0), "100%")
        self.assertEqual(app_module.preview_zoom_label(1.25), "125%")

    def test_document_preview_text_keeps_long_files_scrollable_without_loading_everything(self):
        app_module = load_reconstruction_app()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_report.md"
            path.write_text("\n".join(f"line {idx}" for idx in range(12)), encoding="utf-8")

            text = app_module.document_preview_text(path, max_lines=5)

        self.assertIn("line 0", text)
        self.assertIn("line 4", text)
        self.assertNotIn("line 5", text)
        self.assertIn("showing first 5 lines of 12", text)

    def test_pipeline_report_exports_run_specific_teaching_files(self):
        from pipeline_report import generate_pipeline_report

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = {
                "software": "ClassRoomSTORM Reconstruction V1",
                "video_name": "blink.mp4",
                "frames_processed": 200,
                "threshold_quantile": 0.995,
                "blinker_mode_requested": "auto",
                "blinker_mode_resolved": "multiple",
                "processing_backend_requested": "auto_cpu_gpu_acceleration",
                "processing_backend_used": "gpu_threshold_cpu_components",
                "worker_count": 15,
                "cpu_count": 16,
                "gpu_device": "NVIDIA test GPU",
                "crop_enabled": True,
                "crop": {"x": 4, "y": 5, "width": 40, "height": 30},
                "cumulative_frames": [20, 60, 120, 200],
                "localizations": 4000,
                "truth_used_for_reconstruction": False,
            }

            outputs = generate_pipeline_report(out, summary)

            report_text = (out / "pipeline_report.md").read_text(encoding="utf-8")
            html_text = (out / "pipeline_report.html").read_text(encoding="utf-8")
            extra = out / "extra_files"
            dot_text = (extra / "pipeline_flowchart.dot").read_text(encoding="utf-8")
            svg_text = (out / "pipeline_flowchart.svg").read_text(encoding="utf-8")
            equation_dot_text = (extra / "pipeline_equation_map.dot").read_text(encoding="utf-8")
            equation_svg_text = (out / "pipeline_equation_map.svg").read_text(encoding="utf-8")
            flowchart_size = (extra / "pipeline_flowchart.png").stat().st_size
            equation_map_size = (extra / "pipeline_equation_map.png").stat().st_size

        self.assertTrue(outputs["markdown"].name.endswith("pipeline_report.md"))
        self.assertTrue(outputs["html"].name.endswith("pipeline_report.html"))
        self.assertTrue(outputs["flowchart"].name.endswith("pipeline_flowchart.svg"))
        self.assertEqual(outputs["flowchart_dot"].parent.name, "extra_files")
        self.assertEqual(outputs["flowchart_png"].parent.name, "extra_files")
        self.assertTrue(outputs["equation_map"].name.endswith("pipeline_equation_map.svg"))
        self.assertEqual(outputs["equation_map_dot"].parent.name, "extra_files")
        self.assertEqual(outputs["equation_map_png"].parent.name, "extra_files")
        self.assertIn("digraph", dot_text)
        self.assertIn("rankdir=TB", dot_text)
        self.assertIn("Backend", dot_text)
        self.assertIn("digraph", equation_dot_text)
        self.assertIn("Equation Map", equation_dot_text)
        self.assertGreater(flowchart_size, 15000)
        self.assertGreater(equation_map_size, 15000)
        self.assertTrue("<svg" in svg_text or "Graphviz dot executable not available" in svg_text)
        self.assertTrue("<svg" in equation_svg_text or "Graphviz dot executable not available" in equation_svg_text)
        self.assertIn("q = 0.995", report_text)
        self.assertIn("gpu_threshold_cpu_components", report_text)
        self.assertIn("_component_localizations_from_mask", report_text)
        self.assertIn("x=4, y=5, width=40, height=30", report_text)
        self.assertIn("Centroid", html_text)

    def test_reconstruction_crop_changes_processed_region(self):
        from reconstruction import reconstruct_frames
        from simulation import BlinkerPoint, render_gaussian_frame

        frame = render_gaussian_frame(
            [BlinkerPoint(x_px=30.0, y_px=28.0)],
            (70, 70),
            sigma_px=2.0,
            photons=4000,
            background=1,
            read_noise=0,
            seed=20,
        )
        result = reconstruct_frames([frame], crop=(20, 18, 30, 30))

        self.assertEqual(result.raw_mean.shape, (30, 30))
        self.assertEqual(result.summary["crop"], {"x": 20, "y": 18, "width": 30, "height": 30})
        self.assertLess(abs(result.localizations[0]["x_px"] - 10.0), 1.0)
        self.assertLess(abs(result.localizations[0]["y_px"] - 10.0), 1.0)

    def test_gui_crop_preview_clips_to_loaded_frame(self):
        app_module = load_reconstruction_app()

        self.assertEqual(app_module.clip_crop_to_shape((8, 9, 20, 20), (15, 18)), (8, 9, 10, 6))
        self.assertIsNone(app_module.clip_crop_to_shape((18, 9, 20, 20), (15, 18)))
        self.assertIsNone(app_module.clip_crop_to_shape((2, 2, 0, 8), (15, 18)))

    def test_gui_result_catalog_groups_outputs_for_browser(self):
        app_module = load_reconstruction_app()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in (
                "raw_mean.png",
                "superres.png",
                "raw_mean.npy",
                "superres.npy",
                "summary.json",
                "pipeline_flowchart.svg",
                "pipeline_equation_map.svg",
                "localization_scatter.png",
                "plot_density_map.png",
                "plot_raw_localization_overlay.png",
                "plot_line_profile.png",
            ):
                (tmp_path / name).write_text("placeholder", encoding="utf-8")

            catalog = app_module.result_catalog(tmp_path)

        self.assertEqual([section["key"] for section in catalog], ["main", "diagnostics", "progress", "teaching", "data"])
        main_items = catalog[0]["items"]
        diagnostic_items = catalog[1]["items"]
        teaching_items = catalog[3]["items"]
        data_items = catalog[4]["items"]
        self.assertEqual([item["label"] for item in main_items], ["Raw Mean", "Super-Resolution", "Linked Profiles"])
        self.assertEqual(main_items[-1]["kind"], "linked_profile")
        self.assertEqual([item["label"] for item in diagnostic_items], ["Localization Scatter", "Density Map", "Raw Overlay"])
        self.assertEqual([item["kind"] for item in teaching_items], ["svg", "svg"])
        self.assertEqual(data_items[0]["label"], "Summary JSON")
        self.assertTrue(data_items[0]["path"].name.endswith("summary.json"))

    def test_gui_result_catalog_can_hide_existing_outputs_until_allowed(self):
        app_module = load_reconstruction_app()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "raw_mean.png").write_text("old result", encoding="utf-8")

            catalog = app_module.result_catalog(tmp_path, include_existing=False)

        self.assertEqual([section["key"] for section in catalog], ["main", "diagnostics", "progress", "teaching", "data"])
        self.assertTrue(all(section["items"] == [] for section in catalog))

    def test_virtual_result_catalog_groups_generated_outputs(self):
        app_module = load_virtual_app()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in ("pattern_mask.png", "preview_frame.png", "metadata.json", "truth.csv", "blinking_video.mp4"):
                (tmp_path / name).write_text("placeholder", encoding="utf-8")

            catalog = app_module.virtual_result_catalog(tmp_path)

        self.assertEqual([section["key"] for section in catalog], ["source", "sample", "files"])
        self.assertEqual(catalog[0]["items"][0]["label"], "Pattern Mask")
        self.assertEqual(catalog[1]["items"][0]["label"], "Sample Blink Frame")
        self.assertEqual([item["label"] for item in catalog[2]["items"]], ["Blinking Video", "Truth CSV", "Metadata JSON"])
        self.assertEqual(catalog[2]["items"][0]["kind"], "video")

    def test_virtual_result_catalog_can_hide_existing_outputs_until_allowed(self):
        app_module = load_virtual_app()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "preview_frame.png").write_text("old result", encoding="utf-8")

            catalog = app_module.virtual_result_catalog(tmp_path, include_existing=False)

        self.assertEqual([section["key"] for section in catalog], ["source", "sample", "files"])
        self.assertTrue(all(section["items"] == [] for section in catalog))

    def test_virtual_default_output_folder_names_are_timestamped_and_labeled(self):
        app_module = load_virtual_app()

        stamp = "2026-05-12_153022"

        self.assertEqual(app_module.virtual_default_output_dir("logo", "ignored", stamp).name, f"ClassRoomSTORM_VirtualExperiment_logo_{stamp}")
        self.assertEqual(app_module.virtual_default_output_dir("text", "ABBE", stamp).name, f"ClassRoomSTORM_VirtualExperiment_ABBE_{stamp}")
        self.assertEqual(app_module.virtual_default_output_dir("dots", "", stamp).name, f"ClassRoomSTORM_VirtualExperiment_dot_{stamp}")
        self.assertEqual(app_module.virtual_default_output_dir("text", "A B/B:E", stamp).name, f"ClassRoomSTORM_VirtualExperiment_A_B_B_E_{stamp}")

    def test_reconstruction_default_output_folder_names_are_timestamped_and_labeled(self):
        app_module = load_reconstruction_app()

        stamp = "2026-05-12_153300"

        self.assertEqual(app_module.reconstruction_default_output_dir("", stamp).name, f"ClassRoomSTORM_Reconstruction_{stamp}")
        self.assertEqual(app_module.reconstruction_default_output_dir("ABBE loop", stamp).name, f"ClassRoomSTORM_Reconstruction_ABBE_loop_{stamp}")

    def test_workflow_apps_expose_back_to_studio_launcher_path(self):
        virtual_module = load_virtual_app()
        recon_module = load_reconstruction_app()

        self.assertEqual(virtual_module.studio_launcher_script().name, "ClassRoomSTORM_Studio.py")
        self.assertEqual(recon_module.studio_launcher_script().name, "ClassRoomSTORM_Studio.py")
        self.assertTrue(virtual_module.studio_launcher_script().exists())
        self.assertTrue(recon_module.studio_launcher_script().exists())

    def test_studio_launcher_exposes_v11_workflows_and_clean_equations(self):
        app_module = load_studio_app()

        workflows = app_module.studio_workflows()
        equations = app_module.equation_fragments()
        reader_links = app_module.studio_reader_links()

        self.assertEqual(app_module.APP_VERSION, "V1.2")
        self.assertEqual([item["key"] for item in workflows], ["virtual", "reconstruction"])
        self.assertTrue(workflows[0]["script"].name.endswith("ClassRoomSTORM_VirtualExperiment.py"))
        self.assertTrue(workflows[1]["script"].name.endswith("ClassRoomSTORM_Reconstruction.py"))
        self.assertEqual([item["label"] for item in reader_links], ["Help", "Theory"])
        self.assertTrue(reader_links[0]["path"].name.endswith("ClassRoomSTORM_Help.pdf"))
        self.assertTrue(reader_links[1]["path"].name.endswith("ClassRoomSTORM_Theory.pdf"))
        self.assertNotIn("fallback_path", reader_links[0])
        self.assertNotIn("fallback_path", reader_links[1])
        self.assertIn("Developed by Dr. Awanish Pratap Singh", app_module.studio_footer_text())
        self.assertIn("University of Lübeck", app_module.studio_footer_text())
        self.assertIn(r"$d \approx \frac{\lambda}{2\,\mathrm{NA}}$", equations)
        self.assertIn(r"$k_{\max}=\frac{2\,\mathrm{NA}}{\lambda}$", equations)
        self.assertIn(r"$y_i \sim \mathrm{Poisson}(\mu_i)$", equations)
        self.assertTrue(all(text.startswith("$") and text.endswith("$") for text in equations))
        self.assertTrue(any(r"\frac" in text for text in equations))
        self.assertTrue(any(r"\partial" in text for text in equations))
        self.assertFalse(any("\ufffd" in text or "\u00e2" in text or "\u00ce" in text for text in equations))

    def test_truth_comparison_matches_localizations_within_the_same_frame(self):
        from truth_comparison import nearest_truth_errors

        localizations = [
            {"frame": 0, "x_px": 10.0, "y_px": 10.0},
            {"frame": 1, "x_px": 90.0, "y_px": 90.0},
        ]
        truth_rows = [
            {"frame": "0", "x_px": "12", "y_px": "10", "active": "1"},
            {"frame": "1", "x_px": "80", "y_px": "90", "active": "1"},
            {"frame": "2", "x_px": "10", "y_px": "10", "active": "1"},
            {"frame": "2", "x_px": "90", "y_px": "90", "active": "1"},
        ]

        self.assertEqual(nearest_truth_errors(localizations, truth_rows), [2.0, 10.0])

    def test_studio_mathtext_renderer_returns_transparent_qimage(self):
        app_module = load_studio_app()

        image = app_module.render_equation_image(r"$d \approx \lambda/(2\,\mathrm{NA})$", font_size=18)

        self.assertFalse(image.isNull())
        self.assertGreater(image.width(), 20)
        self.assertGreater(image.height(), 10)

    def test_studio_workflow_transition_is_short_fade(self):
        app_module = load_studio_app()

        self.assertEqual(app_module.STUDIO_TRANSITION_MS, 650)

    def test_studio_reader_display_path_prefers_pdf_then_tex(self):
        app_module = load_studio_app()

        missing_pdf = ROOT / "docs" / "missing_reader.pdf"
        tex_file = ROOT / "docs" / "latex" / "ClassRoomSTORM_Help.tex"
        reader = {"path": missing_pdf, "fallback_path": tex_file}

        self.assertEqual(app_module.reader_display_path(reader), tex_file)

    def test_studio_window_has_opacity_transition_effect(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        app_module = load_studio_app()
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        window = app_module.StudioWindow()

        self.assertEqual(window.overlay_effect.opacity(), 1.0)
        self.assertIsNone(window.pending_workflow_script)
        self.assertTrue(hasattr(window, "begin_workflow_transition"))
        app.processEvents()

    def test_studio_mathtext_uses_computer_modern_style(self):
        app_module = load_studio_app()

        rc = app_module.studio_mathtext_rc()

        self.assertEqual(rc["mathtext.fontset"], "cm")
        self.assertEqual(rc["font.family"], "serif")
        self.assertEqual(rc["mathtext.rm"], "serif")

    def test_studio_equation_font_sizes_stay_subtle(self):
        app_module = load_studio_app()

        sizes = app_module.studio_equation_font_sizes()

        self.assertEqual(sizes, (10, 12, 14, 16))
        self.assertLessEqual(max(sizes), 16)

    def test_studio_mp4_reference_style_uses_slow_motion_and_large_psf_halos(self):
        app_module = load_studio_app()

        style = app_module.studio_reference_style()

        self.assertEqual(style["background_base_colors"], ((0, 0, 0), (5, 6, 10), (0, 0, 0)))
        self.assertEqual(style["equation_drift_px"], (16.0, 9.0))
        self.assertEqual(style["equation_speed_range"], (0.018, 0.055))
        self.assertEqual(style["equation_zoom_range"], (0.86, 1.08))
        self.assertEqual(style["equation_max_width_fraction"], 0.34)
        self.assertGreaterEqual(style["psf_halo_multiplier"], 4.2)
        self.assertEqual(style["background_bottom_right_haze"], (18, 20, 32, 46))

    def test_studio_equation_layout_slots_are_evenly_distributed_outside_center(self):
        app_module = load_studio_app()

        slots = app_module.studio_equation_layout_slots()

        self.assertGreaterEqual(len(slots), len(app_module.equation_fragments()))
        self.assertEqual(len(slots), len(set(slots)))
        for x, y in slots:
            self.assertGreaterEqual(x, 0.02)
            self.assertLessEqual(x, 0.80)
            self.assertGreaterEqual(y, 0.04)
            self.assertLessEqual(y, 0.92)
            self.assertFalse(0.25 < x < 0.62 and 0.24 < y < 0.76)

    def test_studio_equation_rect_is_clamped_inside_window(self):
        app_module = load_studio_app()

        x, y, width, height = app_module.clamped_equation_rect(1230, -25, 420, 80, 1280, 720, margin=18)

        self.assertGreaterEqual(x, 18)
        self.assertGreaterEqual(y, 18)
        self.assertLessEqual(x + width, 1280 - 18)
        self.assertLessEqual(y + height, 720 - 18)

    def test_auto_blinker_mode_detects_multiple_emitters(self):
        from reconstruction import reconstruct_frames
        from simulation import BlinkerPoint, render_gaussian_frame

        frame = render_gaussian_frame(
            [
                BlinkerPoint(x_px=20.0, y_px=20.0, emitter_id=0),
                BlinkerPoint(x_px=48.0, y_px=45.0, emitter_id=1),
            ],
            (80, 80),
            sigma_px=2.0,
            photons=5000,
            background=1,
            read_noise=0,
            seed=33,
        )
        result = reconstruct_frames([frame], blinker_mode="auto", threshold_quantile=0.99)

        self.assertEqual(result.summary["blinker_mode_resolved"], "multiple")
        self.assertGreaterEqual(len(result.localizations), 2)

    def test_reconstruction_rendering_spreads_localizations_for_visibility(self):
        from reconstruction import render_localization_image

        image = render_localization_image([{"x_px": 10.0, "y_px": 10.0}], (25, 25), sigma_px=1.5)

        self.assertGreater(image[10, 10], 0)
        self.assertGreater(image[10, 11], 0)
        self.assertGreater(image.sum(), image[10, 10])

    def test_linked_line_profiles_clamp_click_and_extract_x_y_profiles(self):
        from profile_tools import linked_line_profiles

        raw = np.arange(12, dtype=float).reshape(3, 4)
        recon = np.arange(20, dtype=float).reshape(4, 5)

        profiles = linked_line_profiles(raw, recon, x_px=99, y_px=-5)

        self.assertEqual(profiles["point"], {"x_px": 3, "y_px": 0})
        np.testing.assert_allclose(profiles["raw_x"], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        np.testing.assert_allclose(profiles["recon_x"], [0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(profiles["raw_y"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(profiles["recon_y"], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])

    def test_linked_profile_series_use_raw_red_and_reconstruction_blue(self):
        from profile_tools import linked_profile_series

        series = linked_profile_series()

        self.assertEqual([item["key"] for item in series], ["raw", "reconstruction"])
        self.assertEqual([item["label"] for item in series], ["Raw mean", "Reconstruction"])
        self.assertEqual(series[0]["color"], "#c62828")
        self.assertEqual(series[1]["color"], "#1565c0")

    def test_plot_exports_do_not_require_working_matplotlib(self):
        from plotting import save_localization_scatter, save_side_by_side

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            left = np.zeros((20, 20), dtype=float)
            right = np.eye(20, dtype=float)
            side = save_side_by_side(tmp_path / "side.png", left, right, "left", "right")
            scatter = save_localization_scatter(
                tmp_path / "scatter.png",
                [{"x_px": 4.0, "y_px": 5.0}, {"x_px": 15.0, "y_px": 12.0}],
            )

            self.assertTrue(side.exists())
            self.assertGreater(side.stat().st_size, 0)
            self.assertTrue(scatter.exists())
            self.assertGreater(scatter.stat().st_size, 0)

    def test_publication_plot_bundle_exports(self):
        from plotting import (
            cumulative_checkpoints,
            cumulative_frame_panels,
            save_density_map,
            save_cumulative_reconstruction_panel,
            save_line_profile,
            save_raw_localization_overlay,
            save_superres_png,
        )

        localizations = [
            {"frame": 0, "x_px": 5.0, "y_px": 6.0, "intensity": 100.0},
            {"frame": 1, "x_px": 8.0, "y_px": 10.0, "intensity": 150.0},
            {"frame": 1, "x_px": 8.5, "y_px": 10.5, "intensity": 170.0},
        ]
        self.assertEqual(cumulative_checkpoints(665), [66, 200, 399, 665])
        self.assertEqual(cumulative_checkpoints(200), [20, 60, 120, 200])
        self.assertEqual(cumulative_checkpoints(80), [8, 24, 48, 80])
        self.assertEqual(cumulative_checkpoints(665, manual=[300, 50, 665, 150]), [50, 150, 300, 665])

        multi_locs = [
            {"frame": 0, "x_px": 1.0, "y_px": 1.0},
            {"frame": 0, "x_px": 2.0, "y_px": 2.0},
            {"frame": 1, "x_px": 3.0, "y_px": 3.0},
            {"frame": 4, "x_px": 4.0, "y_px": 4.0},
            {"frame": 9, "x_px": 5.0, "y_px": 5.0},
        ]
        panels = cumulative_frame_panels(multi_locs, total_frames=10)
        self.assertEqual([panel["frame_cutoff"] for panel in panels], [1, 3, 6, 10])
        self.assertEqual([panel["label"] for panel in panels], ["1 frame", "3 frames", "6 frames", "All frames"])
        self.assertEqual([len(panel["localizations"]) for panel in panels], [2, 3, 4, 5])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = np.arange(400, dtype=float).reshape(20, 20)
            recon = np.eye(20, dtype=float)
            files = [
                save_superres_png(tmp_path / "superres.png", recon),
                save_raw_localization_overlay(tmp_path / "plot_raw_localization_overlay.png", raw, localizations),
                save_cumulative_reconstruction_panel(tmp_path / "plot_cumulative_reconstruction.png", localizations, (20, 20), manual_checkpoints=[1, 2, 3, 3]),
                save_line_profile(tmp_path / "plot_line_profile.png", raw, recon),
                save_density_map(tmp_path / "plot_density_map.png", localizations, (20, 20)),
            ]

            for path in files:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
