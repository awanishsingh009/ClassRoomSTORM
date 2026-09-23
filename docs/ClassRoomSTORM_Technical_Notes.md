# ClassRoomSTORM Technical Notes

## Design Scope

ClassRoomSTORM V1.2 is a teaching tool for stochastic blinking and localization-based reconstruction. It is intended for classroom laboratories and paper supplements, not as a replacement for specialized research localization packages.

## Pixel-First Reconstruction

The reconstruction is performed in pixel units. A video may represent microscopy, LED-scale blinking, astronomical-style blinking, or simulation. Optional calibration is applied to exported coordinates. `x_px/y_px` are crop-local, `x_full_px/y_full_px` use the original frame origin, and `x_calibrated/y_calibrated` multiply full-frame coordinates by the supplied unit-per-pixel scale. Images and profiles retain pixel axes.

## Reconstruction Pipeline

1. Read video frames.
2. Optionally crop the field of view.
3. Threshold each frame to identify active blinking regions.
4. Resolve blinker mode: auto, single, or multiple.
5. Localize bright regions using centroid-style localization.
6. Render localizations into a super-resolution image.
7. Export images, plots, CSV, JSON, and teaching pipeline files.

## Virtual Experiment Logic

The virtual experiment converts a dot pattern, text mask, or logo mask into source points. Each frame activates a random subset of source points and renders them as Gaussian-like spots with configurable sigma, brightness, background, and read noise.

For logos and edge emitters, the simulator adds adaptive padding so large spots are not clipped at the image boundary. For text, V1 uses a fixed 16 x 16 LED-style alphabet by default.

## Truth Data Rule

The virtual experiment exports `truth.csv`, but reconstruction must use only the video first. Truth data is used only afterward for comparison. This preserves the teaching logic and prevents the reconstruction from being guided by the answer key.

## Processing Backends

The reconstruction app includes serial CPU, automatic CPU, GPU-only, Auto CPU+GPU, and experimental Auto CPU+GPU acceleration options. NVIDIA CUDA support uses optional PyTorch installation. AMD, Intel GPU, and Apple Silicon acceleration are not implemented in V1; those systems should use CPU or fallback behavior.

## Pipeline Figure Dependencies

ClassRoomSTORM writes `pipeline_report.md`, `pipeline_report.html`, DOT files, PNG figures, and SVG figures after reconstruction. Graphviz is optional. When the `dot` executable is available, the app uses it for the flowchart and equation-map SVG/PNG outputs. When Graphviz is unavailable, V1 falls back to Matplotlib-generated PNG figures and records that Graphviz was not available in the SVG placeholder.

## Limitations

- Strongly overlapping blinkers can be difficult to separate.
- Very noisy, underexposed, or saturated videos may reconstruct poorly.
- The current localization method is transparent and teachable, but simpler than advanced Gaussian-fitting research software.
- GPU behavior is experimental and should be reported from `summary.json` rather than assumed.

## Validation and display contracts

- Optional `frame_median` background correction subtracts the cropped-frame median and clips negative weights to zero. It assumes sparse emitters; it is not an annular estimate or a camera-specific noise model. Raw means remain uncorrected.
- Blank/constant frames produce no detections. Nonfinite, empty, non-grayscale or inconsistent input frames fail before export. Threshold quantiles must be in (0, 1].
- Gaussian display kernels use fractional coordinates, fixed sigma 1.5 camera pixels, and unit sum per localization. Kernels clipped at the boundary are renormalized; display centroids near edges can consequently shift. The render is neither a photon-count map nor a resolution/uncertainty measurement.
- Truth validation uses frame-aware maximum-cardinality one-to-one matching within a stated radius; candidates are visited in distance order. This maximizes match count, not globally minimal total assignment distance. Truth outside the processed crop or frame range is excluded. Unframed truth denotes a static pattern repeated in each evaluated frame. Undefined precision/recall are JSON null.
- The input SHA-256, processing parameters, calibration, crop and backend are saved in summary.json. Existing result folders are not overwritten.
