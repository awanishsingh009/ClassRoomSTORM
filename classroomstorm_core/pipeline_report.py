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
            f"Mean matched error = {truth.get('mean_matched_error_px', truth.get('mean_nearest_truth_error_px', 'not recorded'))} px. "
            f"Precision = {truth.get('precision', 'not recorded')}; recall = {truth.get('recall', 'not recorded')}; F1 = {truth.get('f1', 'not recorded')}. "
            f"Matching radius = {truth.get('max_distance_px', 'not recorded')} px."
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
        - Calibration: {summary.get("calibration", "not recorded")}
        - Background correction: {summary.get("background_mode", "none")}
        - Coordinate convention: {summary.get("coordinate_system", "crop-local pixels")}
        - Frames without localizations: {summary.get("frames_without_localizations", "not recorded")}
        - Input SHA-256: {summary.get("input_sha256", "not recorded")}
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
        The localization intensity I_t below is the nonnegative signal after any selected frame-median subtraction. The raw mean above always uses the original cropped frames.
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
        Each localization is rendered at its fractional pixel position as a small Gaussian visualization kernel, normalized to one count. The camera-grid sampling and fixed display width are not a measured resolution or uncertainty:
        G_i(x,y) = exp(-((x-x_i)^2 + (y-y_i)^2)/(2 sigma_r^2))
        K_i(x,y) = G_i(x,y) / sum_{{u,v in image}} G_i(u,v)
        SR(x,y) = sum_i K_i(x,y)

        Render kernel: {_value(summary, "render_kernel")}

        Code:
        classroomstorm_core/reconstruction.py
        render_localization_image()

        ## Cumulative Reconstruction
        Checkpoints:
        {_value(summary, "cumulative_frames")}

        Default rule:
        10%, 30%, 60%, and 100% of processed frames unless manual checkpoints are provided.

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


def _dot_escape(text: Any) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _diagram_content(summary: dict, equations: bool = False):
    mode = str(summary.get("blinker_mode_resolved", "single"))
    multiple = mode == "multiple"
    background = str(summary.get("background_mode", "none"))
    q = _value(summary, "threshold_quantile")
    if equations:
        labels = [
            "Original cropped frame\nI_t(x,y)",
            "Raw mean\nI_mean = (1/N_frames) sum_t I_t",
            "Localization signal J_t\n" + ("max(I_t - median(I_t), 0)" if background == "frame_median" else "J_t = I_t (no background correction)"),
            f"Threshold and mask\nT_t = Q_q(J_t), q = {q}\nM_t = 1[J_t >= T_t]",
            ("Connected components C_k of M_t" if multiple else "One mask C = M_t") + "\nCentroid: xhat = sum_C x J_t / sum_C J_t\nyhat = sum_C y J_t / sum_C J_t",
            "Normalized display kernels\nG_i = exp(-r_i^2 / (2 sigma_r^2))\nK_i = G_i / sum_image G_i; SR = sum_i K_i",
            "Compare and export\nPositions, images, run parameters",
        ]
    else:
        labels = [
            "Load and crop original frames\n" + str(_value(summary, "video_name")),
            "Average original cropped frames\nRaw mean image",
            "Optional background correction\n" + background,
            f"Quantile threshold q = {q}\nConstant frames yield no localization",
            ("Connected components + centroid per component" if multiple else "Single-emitter thresholded centroid") + "\nBackend: " + str(_value(summary, "processing_backend_used", "serial")),
            "Accumulate positions and render\nUnit-sum Gaussian kernels; sigma = 1.5 px",
            "Compare and export\nImages + CSV + JSON + teaching report",
        ]
    names = ["raw", "mean", "preprocess", "mask", "localize", "render", "export"]
    edges = [("raw","mean"),("raw","preprocess"),("preprocess","mask"),("mask","localize"),("localize","render"),("render","export"),("mean","export")]
    return dict(zip(names, labels)), edges


def _build_diagram_dot(summary: dict, equations: bool = False) -> str:
    nodes, edges = _diagram_content(summary, equations)
    title = "Equation Map For The Selected Reconstruction Path" if equations else "ClassRoomSTORM Reconstruction Pipeline"
    lines = ['digraph ClassRoomSTORM {', 'graph [rankdir=TB, bgcolor="white", pad="0.3", nodesep="0.5", ranksep="0.5", splines=polyline, fontname="Times New Roman", fontsize=18, labelloc=t, label="'+title+'"];', 'node [shape=box, style="rounded,filled", fontname="Times New Roman", fontsize=11, margin="0.15,0.1", color="#0072B2", fillcolor="#F4F9FC", penwidth=1];', 'edge [color="#69717A", arrowsize=0.7, penwidth=1];']
    for name, label in nodes.items():
        colour = "#c1272d" if name == "mean" else "#39775A" if name == "export" else "#0072B2"
        lines.append(f'{name} [label="{_dot_escape(label)}", color="{colour}"];')
    lines.extend(f'{a} -> {b};' for a,b in edges)
    lines.append('{rank=same; mean; preprocess;}')
    lines.append('}')
    return "\n".join(lines) + "\n"


def build_flowchart_dot(summary: dict) -> str:
    return _build_diagram_dot(summary)


def build_equation_map_dot(summary: dict) -> str:
    return _build_diagram_dot(summary, equations=True)


def _save_diagram(path: Path, summary: dict, equations: bool = False) -> None:
    import matplotlib.pyplot as plt
    try:
        from .figure_style import configure
    except ImportError:
        from figure_style import configure
    configure(plt)
    nodes, edges = _diagram_content(summary, equations)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
    ax.set_title("Equation Map" if equations else "Reconstruction pipeline", pad=10)
    positions = {"raw":(.35,.91), "mean":(.80,.91), "preprocess":(.35,.75), "mask":(.35,.59), "localize":(.35,.42), "render":(.35,.24), "export":(.35,.07)}
    artists = {}
    for name,label in nodes.items():
        x,y = positions[name]
        artists[name] = ax.text(x,y,label,ha="center",va="center",fontsize=8.5 if equations else 9,
                bbox=dict(boxstyle="round,pad=.55",facecolor="#F4F9FC",edgecolor="#c1272d" if name=="mean" else "#0072B2",linewidth=.8))
    fig.subplots_adjust(left=.02,right=.98,bottom=.035,top=.94)
    fig.canvas.draw()
    bounds = {name: artist.get_bbox_patch().get_window_extent().transformed(ax.transData.inverted())
              for name, artist in artists.items()}
    for a,b in edges:
        x,y=positions[a];u,v=positions[b]
        if a=="raw" and b=="mean":
            ax.annotate("",xy=(bounds[b].x0,.91),xytext=(bounds[a].x1,.91),arrowprops=dict(arrowstyle="->",color="#69717A",lw=.8))
        elif a=="mean":
            ax.plot([.8,.8,.65],[bounds[a].y0,.07,.07],color="#69717A",lw=.8)
            ax.annotate("",xy=(bounds[b].x1,.07),xytext=(.65,.07),arrowprops=dict(arrowstyle="->",color="#69717A",lw=.8))
        else:
            ax.annotate("",xy=(u,bounds[b].y1),xytext=(x,bounds[a].y0),arrowprops=dict(arrowstyle="->",color="#69717A",lw=.8))
    fig.text(.5,.005,"Display kernel width is not position uncertainty or measured resolution.",ha="center",fontsize=8)
    fig.subplots_adjust(left=.02,right=.98,bottom=.035,top=.94)
    fig.savefig(path,dpi=300)
    # A real vector fallback is available even without Graphviz.
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def _save_flowchart(path: Path, summary: dict) -> None:
    _save_diagram(path, summary)


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
def _save_equation_map(path: Path, summary: dict) -> None:
    _save_diagram(path, summary, equations=True)


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
        shutil.copy2(flowchart_png_path.with_suffix(".svg"), flowchart_path)
    if not _render_dot_graph(equation_map_png_path, equation_map_path, equation_map_dot_path, build_equation_map_dot(summary)):
        _save_equation_map(equation_map_png_path, summary)
        shutil.copy2(equation_map_png_path.with_suffix(".svg"), equation_map_path)
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
