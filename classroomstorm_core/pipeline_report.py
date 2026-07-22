from __future__ import annotations

import html
import inspect
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any


def _value(summary: dict, key: str, default: Any = "not recorded") -> Any:
    value = summary.get(key, default)
    return default if value in (None, "") else value


def _crop_text(summary: dict) -> str:
    if not summary.get("crop_enabled"):
        return "Crop disabled"
    crop = summary.get("crop") or {}
    if isinstance(crop, dict):
        return (
            f"x={crop.get('x', '?')}, y={crop.get('y', '?')}, "
            f"width={crop.get('width', '?')}, height={crop.get('height', '?')}"
        )
    return str(crop)


def _backend_explanation(summary: dict) -> str:
    used = str(summary.get("processing_backend_used", "serial"))
    if used == "serial":
        return "Frames were localized one after another on one CPU worker."
    if used == "cpu_parallel":
        return "Frames were split across CPU workers. Each worker localized separate frames."
    if used == "gpu_torch":
        return "Single-emitter centroid localization ran as a batched PyTorch CUDA operation."
    if used == "gpu_threshold_cpu_components":
        return "GPU created threshold masks in batches; CPU workers performed connected-component localization."
    return "The selected backend used the recorded fallback or experimental path."


def _localization_section(summary: dict) -> str:
    mode = str(summary.get("blinker_mode_resolved", "single"))
    if mode == "multiple":
        return textwrap.dedent(
            """
            Step: Multiple-emitter segmentation and Centroid localization
            What happened:
            Bright pixels were separated into connected components. Each component C_k was treated as one localized blinker.

            Equations:
            C_k = connected components of M_t(x,y)
            x_hat_k = sum_{(x,y) in C_k} x I_t(x,y) / sum_{(x,y) in C_k} I_t(x,y)
            y_hat_k = sum_{(x,y) in C_k} y I_t(x,y) / sum_{(x,y) in C_k} I_t(x,y)

            Code:
            classroomstorm_core/reconstruction.py
            _component_localizations()
            _component_localizations_from_mask()
            _connected_components()
            """
        ).strip()
    return textwrap.dedent(
        """
        Step: Single-emitter centroid localization
        What happened:
        One bright region was localized in each accepted frame.

        Equations:
        x_hat = sum_x,y x I_t(x,y) M_t(x,y) / sum_x,y I_t(x,y) M_t(x,y)
        y_hat = sum_x,y y I_t(x,y) M_t(x,y) / sum_x,y I_t(x,y) M_t(x,y)

        Code:
        classroomstorm_core/reconstruction.py
        _localize_single()
        _centroid_localization()
        """
    ).strip()


def build_pipeline_markdown(summary: dict) -> str:
    timing = summary.get("timing_seconds") or {}
    truth = summary.get("truth_comparison")
    truth_text = "Truth comparison was not run."
    if truth:
        truth_text = (
            f"Truth comparison matched {truth.get('matched_localizations', 'not recorded')} localizations. "
            f"Mean nearest-truth error = {truth.get('mean_nearest_truth_error_px', 'not recorded')} px."
        )

    text = inspect.cleandoc(
        f"""
        # ClassRoomSTORM Reconstruction Teaching Pipeline

        ## Run Values
        - Software: {_value(summary, "software")}
        - Video: {_value(summary, "video_name")}
        - Frames processed: {_value(summary, "frames_processed")}
        - Localizations: {_value(summary, "localizations")}
        - Pixel mode: {_value(summary, "pixel_mode", "pixels")}
        - Crop: {_crop_text(summary)}
        - Blinker mode requested: {_value(summary, "blinker_mode_requested")}
        - Blinker mode resolved: {_value(summary, "blinker_mode_resolved")}
        - Threshold quantile: q = {_value(summary, "threshold_quantile")}
        - Backend requested: {_value(summary, "processing_backend_requested")}
        - Backend used: {_value(summary, "processing_backend_used")}
        - CPU workers: {_value(summary, "worker_count")}
        - CPU count: {_value(summary, "cpu_count")}
        - GPU device: {_value(summary, "gpu_device", "not used")}
        - Backend note: {_value(summary, "gpu_note", "none")}
        - Cumulative frames: {_value(summary, "cumulative_frames")}
        - Localization time: {timing.get("localization", "not recorded")} s
        - Total reconstruction-core time: {timing.get("total_reconstruct_frames", "not recorded")} s

        ## Pipeline
        1. Load video frames from the selected MP4.
        2. Select frame range: N = {_value(summary, "frames_processed")} frames.
        3. Apply crop/preprocessing: {_crop_text(summary)}.
        4. Resolve blinker mode: {_value(summary, "blinker_mode_requested")} -> {_value(summary, "blinker_mode_resolved")}.
        5. Select compute backend: {_value(summary, "processing_backend_requested")} -> {_value(summary, "processing_backend_used")}.
        6. Threshold each frame.
        7. Localize blinkers.
        8. Render the super-resolution image.
        9. Export images, plots, CSV, JSON, and this teaching report.

        ## Image And Threshold Equations
        Raw video frame:
        I_t(x,y)

        Raw mean:
        I_mean(x,y) = (1/N) sum_t I_t(x,y)

        Threshold:
        T_t = quantile(I_t, q), where q = {_value(summary, "threshold_quantile")}

        Binary mask:
        M_t(x,y) = 1 if I_t(x,y) >= T_t else 0

        Code:
        classroomstorm_core/reconstruction.py
        _localize_single()
        _component_localizations()

        ## Localization Equation Map
        {_localization_section(summary)}

        ## Backend Equation / Execution Map
        Backend used: {_value(summary, "processing_backend_used")}

        {_backend_explanation(summary)}

        Code:
        classroomstorm_core/reconstruction.py
        reconstruct_frames()
        _resolve_processing_backend()
        _localize_frames()
        _localize_frames_gpu_single()
        _localize_frames_gpu_threshold_cpu_components()

        ## Super-Resolution Rendering
        Each localization is rendered as a small Gaussian visualization kernel:
        SR(x,y) = sum_i exp(-((x-x_i)^2 + (y-y_i)^2)/(2 sigma_r^2))

        Render kernel: {_value(summary, "render_kernel")}

        Code:
        classroomstorm_core/reconstruction.py
        render_localization_image()

        ## Cumulative Reconstruction
        Checkpoints:
        {_value(summary, "cumulative_frames")}

        Default rule:
        10%, 30%, 60%, and 100% of total localizations unless manual checkpoints are provided.

        Code:
        classroomstorm_core/plotting.py
        save_cumulative_reconstruction_panel()

        ## Truth Comparison
        {truth_text}

        Truth is never used before reconstruction. It is only used after reconstruction for optional validation.

        ## Main Outputs
        - raw_mean.png
        - superres.png
        - raw_vs_superres.png
        - localization_scatter.png
        - plot_density_map.png
        - plot_raw_localization_overlay.png
        - plot_cumulative_reconstruction.png
        - plot_line_profile.png
        - localizations.csv
        - summary.json
        - pipeline_report.md
        - pipeline_report.html
        - pipeline_flowchart.svg
        - pipeline_equation_map.svg
        - extra_files/pipeline_flowchart.dot
        - extra_files/pipeline_flowchart.png
        - extra_files/pipeline_equation_map.dot
        - extra_files/pipeline_equation_map.png
        """
    )
    lines = [line[8:] if line.startswith("        ") else line for line in text.splitlines()]
    return "\n".join(lines).strip() + "\n"


def _write_html(path: Path, markdown_text: str) -> None:
    escaped = html.escape(markdown_text)
    html_text = textwrap.dedent(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>ClassRoomSTORM Reconstruction Teaching Pipeline</title>
          <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.5; max-width: 980px; margin: 32px auto; padding: 0 20px; color: #111827; }}
            pre {{ white-space: pre-wrap; background: #f3f4f6; border: 1px solid #d1d5db; padding: 16px; border-radius: 6px; }}
          </style>
        </head>
        <body>
          <h1>ClassRoomSTORM Reconstruction Teaching Pipeline</h1>
          <pre>{escaped}</pre>
        </body>
        </html>
        """
    ).strip()
    path.write_text(html_text, encoding="utf-8")


def _draw_box(
    ax: Any,
    x: float,
    y: float,
    text: str,
    width: float = 0.18,
    height: float = 0.105,
    facecolor: str = "#ecfeff",
    edgecolor: str = "#0f766e",
    fontsize: float = 9.0,
) -> None:
    import matplotlib.patches as patches

    rect = patches.FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True, transform=ax.transAxes, color="#111827")


def _draw_arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], label: str = "") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#334155", "shrinkA": 6, "shrinkB": 6},
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.025,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#475569",
            transform=ax.transAxes,
        )


def _save_flowchart(path: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.96,
        "ClassRoomSTORM Reconstruction Pipeline",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.925,
        f"Run-specific path: {_value(summary, 'blinker_mode_requested')} -> {_value(summary, 'blinker_mode_resolved')} | "
        f"{_value(summary, 'processing_backend_requested')} -> {_value(summary, 'processing_backend_used')}",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
        transform=ax.transAxes,
    )

    boxes = {
        "input": (0.08, 0.72, f"Input video\n{_value(summary, 'video_name')}", "#e0f2fe", "#0369a1"),
        "frames": (0.25, 0.72, f"Frame selection\nN = {_value(summary, 'frames_processed')}", "#e0f2fe", "#0369a1"),
        "crop": (0.42, 0.72, f"Crop / preprocess\n{_crop_text(summary)}", "#e0f2fe", "#0369a1"),
        "mode": (0.60, 0.72, f"Blinker mode\n{_value(summary, 'blinker_mode_requested')} -> {_value(summary, 'blinker_mode_resolved')}", "#fef3c7", "#b45309"),
        "single": (0.55, 0.49, "Single emitter path\nthreshold + centroid", "#f0fdf4", "#15803d"),
        "multiple": (0.75, 0.49, "Multiple emitter path\nthreshold + components", "#f0fdf4", "#15803d"),
        "backend": (0.38, 0.30, f"Compute backend\n{_value(summary, 'processing_backend_used')}", "#f5f3ff", "#6d28d9"),
        "loc": (0.62, 0.30, "Localization table\nx, y, intensity, frame", "#f0fdf4", "#15803d"),
        "render": (0.81, 0.30, "SR rendering\nGaussian visualization", "#fff7ed", "#c2410c"),
        "export": (0.62, 0.12, "Exports\nimages, CSV, JSON,\nteaching report", "#f8fafc", "#475569"),
    }
    for x, y, text, face, edge in boxes.values():
        _draw_box(ax, x, y, text, facecolor=face, edgecolor=edge, width=0.14 if x in {0.08, 0.25, 0.42} else 0.16)

    _draw_arrow(ax, (0.15, 0.72), (0.18, 0.72))
    _draw_arrow(ax, (0.32, 0.72), (0.35, 0.72))
    _draw_arrow(ax, (0.49, 0.72), (0.52, 0.72))
    _draw_arrow(ax, (0.58, 0.66), (0.55, 0.56), "single")
    _draw_arrow(ax, (0.66, 0.66), (0.75, 0.56), "multiple")
    _draw_arrow(ax, (0.53, 0.43), (0.42, 0.36))
    _draw_arrow(ax, (0.74, 0.43), (0.64, 0.36))
    _draw_arrow(ax, (0.46, 0.30), (0.54, 0.30))
    _draw_arrow(ax, (0.70, 0.30), (0.73, 0.30))
    _draw_arrow(ax, (0.81, 0.23), (0.68, 0.17))
    _draw_arrow(ax, (0.62, 0.23), (0.62, 0.18))

    ax.text(0.08, 0.58, r"$I_t(x,y)$", fontsize=12, color="#0f172a", transform=ax.transAxes)
    ax.text(0.31, 0.58, r"$I_{\mathrm{mean}}=\frac{1}{N}\sum_t I_t$", fontsize=12, color="#0f172a", transform=ax.transAxes)
    ax.text(0.72, 0.63, r"$T_t=Q_q(I_t),\quad M_t=\mathbf{1}[I_t\geq T_t]$", fontsize=12, color="#0f172a", transform=ax.transAxes)
    ax.text(0.11, 0.22, f"CPU workers: {_value(summary, 'worker_count')} / CPU count: {_value(summary, 'cpu_count')}", fontsize=9, color="#475569", transform=ax.transAxes)
    ax.text(0.11, 0.17, f"GPU: {_value(summary, 'gpu_device', 'not used')}", fontsize=9, color="#475569", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _dot_escape(text: Any) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def build_flowchart_dot(summary: dict) -> str:
    mode = str(_value(summary, "blinker_mode_resolved", "single"))
    backend = str(_value(summary, "processing_backend_used", "serial"))
    gpu_label = _value(summary, "gpu_device", "not used")
    localization_label = "Connected components + centroid per component" if mode == "multiple" else "Single centroid localization"
    backend_label = {
        "serial": "Serial CPU",
        "cpu_parallel": f"CPU parallel\\nworkers={_value(summary, 'worker_count')}",
        "gpu_torch": f"GPU torch\\n{gpu_label}",
        "gpu_threshold_cpu_components": f"GPU threshold + CPU components\\nworkers={_value(summary, 'worker_count')}",
    }.get(backend, backend)
    return textwrap.dedent(
        f"""
        digraph ClassRoomSTORMPipeline {{
          graph [
            rankdir=TB,
            bgcolor="white",
            pad="0.35",
            nodesep="0.45",
            ranksep="0.55",
            splines=ortho,
            fontname="Arial",
            label="ClassRoomSTORM Reconstruction Pipeline",
            labelloc=t,
            fontsize=22
          ];
          node [
            shape=box,
            style="rounded,filled",
            fontname="Arial",
            fontsize=12,
            margin="0.11,0.08",
            color="#0f766e",
            fillcolor="#ecfeff",
            penwidth=1.4
          ];
          edge [fontname="Arial", fontsize=10, color="#334155", arrowsize=0.8, penwidth=1.3];

          input [label="Input video\\n{_dot_escape(_value(summary, 'video_name'))}", fillcolor="#e0f2fe", color="#0369a1"];
          frames [label="Frame selection\\nN={_dot_escape(_value(summary, 'frames_processed'))}", fillcolor="#e0f2fe", color="#0369a1"];
          crop [label="Crop / preprocess\\n{_dot_escape(_crop_text(summary))}", fillcolor="#e0f2fe", color="#0369a1"];
          mode [label="Blinker mode decision\\n{_dot_escape(_value(summary, 'blinker_mode_requested'))} -> {_dot_escape(_value(summary, 'blinker_mode_resolved'))}", fillcolor="#fef3c7", color="#b45309"];

          single [label="Single-emitter path\\nthreshold + centroid", fillcolor="#f0fdf4", color="#15803d"];
          multiple [label="Multiple-emitter path\\nthreshold + connected components", fillcolor="#f0fdf4", color="#15803d"];
          backend [label="Backend\\n{_dot_escape(_value(summary, 'processing_backend_requested'))} -> {_dot_escape(backend_label)}", fillcolor="#f5f3ff", color="#6d28d9"];
          localize [label="{_dot_escape(localization_label)}\\nlocalizations={_dot_escape(_value(summary, 'localizations'))}", fillcolor="#f0fdf4", color="#15803d"];
          render [label="Super-resolution rendering\\nGaussian visualization", fillcolor="#fff7ed", color="#c2410c"];
          export [label="Exports\\nPNG plots + CSV + JSON\\nteaching report", fillcolor="#f8fafc", color="#475569"];

          input -> frames -> crop -> mode;
          {{ rank=same; single; multiple; }}
          mode -> single [label="single", tailport=s, headport=n];
          mode -> multiple [label="multiple", tailport=s, headport=n];
          single -> backend;
          multiple -> backend;
          backend -> localize -> render -> export;
        }}
        """
    ).strip() + "\n"


def build_equation_map_dot(summary: dict) -> str:
    mode = str(summary.get("blinker_mode_resolved", "single"))
    if mode == "multiple":
        localization = (
            "C_k = components(M_t)\\n"
            "xhat_k = sum_C x I_t / sum_C I_t\\n"
            "yhat_k = sum_C y I_t / sum_C I_t"
        )
    else:
        localization = (
            "xhat = sum x I_t M_t / sum I_t M_t\\n"
            "yhat = sum y I_t M_t / sum I_t M_t"
        )
    return textwrap.dedent(
        f"""
        digraph ClassRoomSTORMEquationMap {{
          graph [
            rankdir=TB,
            bgcolor="white",
            pad="0.35",
            nodesep="0.45",
            ranksep="0.55",
            splines=ortho,
            fontname="Arial",
            label="Equation Map For The Selected Reconstruction Path",
            labelloc=t,
            fontsize=22
          ];
          node [
            shape=box,
            style="rounded,filled",
            fontname="Arial",
            fontsize=12,
            margin="0.12,0.08",
            color="#0f766e",
            fillcolor="#f8fafc",
            penwidth=1.4
          ];
          edge [fontname="Arial", fontsize=10, color="#334155", arrowsize=0.8, penwidth=1.3];

          raw [label="Raw frame\\nI_t(x,y)", fillcolor="#e0f2fe", color="#0369a1"];
          mean [label="Raw mean\\nI_mean(x,y) = (1/N) sum_t I_t(x,y)", fillcolor="#e0f2fe", color="#0369a1"];
          threshold [label="Threshold\\nT_t = Q_q(I_t)\\nq = {_dot_escape(_value(summary, 'threshold_quantile'))}", fillcolor="#fef3c7", color="#b45309"];
          mask [label="Binary mask\\nM_t(x,y) = 1 if I_t(x,y) >= T_t else 0", fillcolor="#fef3c7", color="#b45309"];
          localize [label="Localization\\n{_dot_escape(localization)}", fillcolor="#f0fdf4", color="#15803d"];
          render [label="SR rendering\\nSR(x,y) = sum_i exp(-((x-x_i)^2+(y-y_i)^2)/(2 sigma_r^2))", fillcolor="#fff7ed", color="#c2410c"];

          raw -> mean;
          raw -> threshold -> mask -> localize -> render;
          {{ rank=same; mean; threshold; }}
        }}
        """
    ).strip() + "\n"


def _save_flowchart_graphviz(png_path: Path, svg_path: Path, dot_path: Path, summary: dict) -> bool:
    dot_text = build_flowchart_dot(summary)
    dot_path.write_text(dot_text, encoding="utf-8")
    dot_exe = shutil.which("dot")
    if dot_exe is None:
        for candidate in (
            Path("C:/Program Files/Graphviz/bin/dot.exe"),
            Path("C:/Program Files (x86)/Graphviz/bin/dot.exe"),
        ):
            if candidate.exists():
                dot_exe = str(candidate)
                break
    if dot_exe is None:
        svg_path.write_text("<!-- Graphviz dot executable not available; PNG rendered with Matplotlib fallback. -->\n", encoding="utf-8")
        return False


def _render_dot_graph(png_path: Path, svg_path: Path, dot_path: Path, dot_text: str) -> bool:
    dot_path.write_text(dot_text, encoding="utf-8")
    dot_exe = shutil.which("dot")
    if dot_exe is None:
        for candidate in (
            Path("C:/Program Files/Graphviz/bin/dot.exe"),
            Path("C:/Program Files (x86)/Graphviz/bin/dot.exe"),
        ):
            if candidate.exists():
                dot_exe = str(candidate)
                break
    if dot_exe is None:
        svg_path.write_text("<!-- Graphviz dot executable not available. -->\n", encoding="utf-8")
        return False
    try:
        subprocess.run([dot_exe, "-Tpng", "-Gdpi=300", str(dot_path), "-o", str(png_path)], check=True, capture_output=True, text=True)
        subprocess.run([dot_exe, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True, capture_output=True, text=True)
        return True
    except Exception as exc:
        svg_path.write_text(f"<!-- Graphviz rendering failed: {html.escape(str(exc))}. -->\n", encoding="utf-8")
        return False
    try:
        subprocess.run([dot_exe, "-Tpng", "-Gdpi=300", str(dot_path), "-o", str(png_path)], check=True, capture_output=True, text=True)
        subprocess.run([dot_exe, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True, capture_output=True, text=True)
        return True
    except Exception as exc:
        svg_path.write_text(f"<!-- Graphviz rendering failed: {html.escape(str(exc))}; PNG rendered with Matplotlib fallback. -->\n", encoding="utf-8")
        return False


def _save_equation_map(path: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    mode = str(summary.get("blinker_mode_resolved", "single"))
    localization_eqs = (
        [
            r"$C_k=\mathrm{components}(M_t)$",
            r"$\hat{x}_k=\frac{\sum_{(x,y)\in C_k}x\,I_t(x,y)}{\sum_{(x,y)\in C_k}I_t(x,y)}$",
            r"$\hat{y}_k=\frac{\sum_{(x,y)\in C_k}y\,I_t(x,y)}{\sum_{(x,y)\in C_k}I_t(x,y)}$",
        ]
        if mode == "multiple"
        else [
            r"$\hat{x}=\frac{\sum_{x,y}x\,I_t(x,y)M_t(x,y)}{\sum_{x,y}I_t(x,y)M_t(x,y)}$",
            r"$\hat{y}=\frac{\sum_{x,y}y\,I_t(x,y)M_t(x,y)}{\sum_{x,y}I_t(x,y)M_t(x,y)}$",
        ]
    )
    blocks = [
        ("Raw frame", [r"$I_t(x,y)$"]),
        ("Raw mean", [r"$I_{\mathrm{mean}}(x,y)=\frac{1}{N}\sum_{t=1}^{N}I_t(x,y)$"]),
        ("Threshold", [r"$T_t=Q_q(I_t)$", rf"$q={_value(summary, 'threshold_quantile')}$"]),
        ("Binary mask", [r"$M_t(x,y)=\mathbf{1}\left[I_t(x,y)\geq T_t\right]$"]),
        ("Localization", localization_eqs),
        ("SR rendering", [r"$SR(x,y)=\sum_i\exp\left(-\frac{(x-x_i)^2+(y-y_i)^2}{2\sigma_r^2}\right)$"]),
    ]
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.96,
        "Equation Map For The Selected Reconstruction Path",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.925,
        f"Mode: {_value(summary, 'blinker_mode_resolved')} | Backend: {_value(summary, 'processing_backend_used')}",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
        color="#475569",
    )
    for idx, (title, body) in enumerate(blocks):
        col = idx % 2
        row = idx // 2
        x0 = 0.06 + col * 0.48
        y0 = 0.72 - row * 0.235
        rect = patches.FancyBboxPatch(
            (x0, y0),
            0.40,
            0.17,
            boxstyle="round,pad=0.018",
            linewidth=1.2,
            edgecolor="#0f766e",
            facecolor="#f8fafc",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x0 + 0.02, y0 + 0.135, title, fontsize=11, fontweight="bold", color="#0f766e", transform=ax.transAxes)
        for line_idx, equation in enumerate(body):
            ax.text(x0 + 0.02, y0 + 0.09 - line_idx * 0.05, equation, fontsize=12, color="#111827", transform=ax.transAxes)
        if idx < len(blocks) - 1:
            if col == 0:
                _draw_arrow(ax, (x0 + 0.41, y0 + 0.085), (x0 + 0.48, y0 + 0.085))
            else:
                _draw_arrow(ax, (x0 + 0.20, y0 - 0.01), (0.26, y0 - 0.065))
    ax.text(
        0.06,
        0.06,
        "Math is rendered with Matplotlib mathtext, so no external LaTeX installation is required.",
        fontsize=9,
        color="#475569",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def generate_pipeline_report(output_dir: str | Path, summary: dict) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    markdown_path = out / "pipeline_report.md"
    html_path = out / "pipeline_report.html"
    flowchart_path = out / "pipeline_flowchart.svg"
    equation_map_path = out / "pipeline_equation_map.svg"
    extra = out / "extra_files"
    extra.mkdir(parents=True, exist_ok=True)
    flowchart_png_path = extra / "pipeline_flowchart.png"
    flowchart_dot_path = extra / "pipeline_flowchart.dot"
    equation_map_png_path = extra / "pipeline_equation_map.png"
    equation_map_dot_path = extra / "pipeline_equation_map.dot"

    markdown_text = build_pipeline_markdown(summary)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    _write_html(html_path, markdown_text)
    if not _render_dot_graph(flowchart_png_path, flowchart_path, flowchart_dot_path, build_flowchart_dot(summary)):
        _save_flowchart(flowchart_png_path, summary)
        flowchart_path.write_text("<!-- Graphviz dot executable not available; PNG rendered with Matplotlib fallback. -->\n", encoding="utf-8")
    if not _render_dot_graph(equation_map_png_path, equation_map_path, equation_map_dot_path, build_equation_map_dot(summary)):
        _save_equation_map(equation_map_png_path, summary)
        equation_map_path.write_text("<!-- Graphviz dot executable not available; PNG rendered with Matplotlib fallback. -->\n", encoding="utf-8")
    return {
        "markdown": markdown_path,
        "html": html_path,
        "flowchart": flowchart_path,
        "flowchart_png": flowchart_png_path,
        "flowchart_dot": flowchart_dot_path,
        "equation_map": equation_map_path,
        "equation_map_png": equation_map_png_path,
        "equation_map_dot": equation_map_dot_path,
    }
