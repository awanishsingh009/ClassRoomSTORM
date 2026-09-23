"""Reproduce a small virtual-to-reconstruction check and a real-video smoke run.

Outputs are created only in a new or empty directory. These checks validate
software behavior, not research-grade SMLM accuracy or experimental resolution.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def load_app(repo, name):
    spec = importlib.util.spec_from_file_location(name, repo / "apps" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("output must be a new or empty directory")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    repo = Path(__file__).resolve().parents[1]
    virtual = load_app(repo, "ClassRoomSTORM_VirtualExperiment")
    recon = load_app(repo, "ClassRoomSTORM_Reconstruction")
    virtual.generate_dataset(out / "virtual", pattern="dots", dot_mode="single", frames=20,
                             blinkers_per_frame=1, sigma_px=2, photons=20000,
                             background=1, read_noise=0.1, worker_count=1, seed=2026)
    video = out / "virtual/blinking_video.mp4"
    info = recon.read_video_info(video)
    summary = recon.run_reconstruction(video, out / "virtual_reconstruction",
        crop=(3, 4, info["width"] - 6, info["height"] - 8), compare_truth=True,
        pixel_size=0.25, unit="mm", background_mode="frame_median", blinker_mode="single")
    truth = summary["truth_comparison"]
    if truth["f1"] != 1 or truth["mean_matched_error_px"] >= 0.5:
        raise SystemExit(f"Synthetic validation failed: {truth}")
    real = recon.run_reconstruction(repo / "assets/experimental_led_matrix_video_2023-09-07.mp4",
                                   out / "experimental_reconstruction", max_frames=40, blinker_mode="auto")
    result = {"synthetic": truth, "experimental_smoke": {"frames": real["frames_processed"],
              "localizations": real["localizations"]},
              "scope": "Seeded single-emitter synthetic check and 40-frame experimental smoke test; no experimental accuracy claim."}
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
