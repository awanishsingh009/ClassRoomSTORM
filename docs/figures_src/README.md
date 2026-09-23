# Scientific figure sources

The 2026-09-21 revision uses physical meaning to choose presentation. DFG R1
WP1.1–WP4.2 and overall V9 informed spacing, mathematical typography and the
restrained palette; their layouts were not copied as a compulsory template.

## Rebuild

Use Python 3.11 or newer, the optional `requirements-figures.txt`, a working
LaTeX installation, and Poppler (`pdftoppm`, `pdftocairo`) on PATH. From the
ClassRoomSTORM_V1 directory:

```powershell
python -m pip install -r requirements-figures.txt
python docs/figures_src/make_screenshots.py
python tools/build_figures.py
python tools/build_docs.py
```

In a standalone software clone, these commands build the two manuals and their
20 figure entries. The separate research workspace is optional. When it contains
`../manuscript_writing/ClassRoomSTORM_AJP.tex`, the tools also regenerate and map
the manuscript figures. In that workspace, compile the manuscript with pdflatex
into its `build/` directory, repeating until references stabilize, and copy
the resulting PDF beside the source. Then run
`python tools/audit_figure_typography.py` from the application directory.
This checks the actual compiled pages, figure-number mappings, and insertion
widths. `--gallery-only` on `build_figures.py` refreshes the review without
recomputing the scientific masters.

`reference/Setup_SuperResolution.JPG` preserves the supplied apparatus image
used by `tikz/P6_apparatus.tex`. No historical workspace folder is needed to
compile that schematic. The original reference image remains unchanged.

`make_scientific_figures.py` owns numerical figures, `figure_data.py` owns the
physical examples, and `tikz/` owns native schematic sources. Older generator
entry points forward to the consolidated build. `style.py` defines final
printed dimensions and plotting typography. Runtime diagnostics use
`classroomstorm_core/figure_style.py` and do not require LaTeX or SciPy.

## Figure contract

- Full-width masters are 162 mm wide. Narrow figures are explicitly sized
  at 70%, 72%, 80%, 86% or 90% of this width. Avoid arbitrary shrinking.
- Latin Modern serif text and mathematics. Main labels and panel titles are
  9 pt; ticks, legends and notes are 8.5 pt, at the printed size. This stays
  within 3 pt of the 11 pt manual body and below the 10 pt manuscript body.
  Mathematical subscripts/superscripts retain their natural smaller size.
  Original raster UI screenshots retain the application's native typography;
  the scientific-label size contract does not apply to text embedded in them.
- Panel markers `(a), (b), ...` are attached to left-aligned titles. Single-panel
  plots have no redundant `(a)`. Thin axes and restrained rules sit on white
  backgrounds. A build-time check rejects labels outside the fixed canvas.
- Diagram boxes align to a shared grid. Directional arrows connect box edges;
  the raw-mean branch stays separate from background subtraction/localization.
- Public figure numbers come from document order and LaTeX labels, not internal
  asset filenames. `figure_locations.json` maps the 24 review entries to their
  31 appearances across the manuals and manuscript. The gallery shows the
  appropriate document/figure number and all shared uses.
- Raw signal: dark red, dashed in comparisons. Rendered positions: blue,
  solid. Truth: green guides/plus markers. Mechanism or selected region:
  orange. Color is supplemented by line style or marker shape.
- Camera-like fields use grayscale with declared normalization. The Airy
  ring image uses a labelled logarithmic scale. Quantitative profiles remain
  linear unless an axis explicitly declares logarithmic scaling.
- The source mask, optical PSF, detector integration, localization estimator,
  and display kernel are separate concepts. Display width is not uncertainty.
- Numerical teaching panels use upward-positive displayed y coordinates;
  exported application arrays retain their camera-row convention. Comparisons
  within each figure use one consistent coordinate system and field of view.
- Diagrams are editable TikZ; plots are editable Python plus exported CSV.
  PDF is the publication master, SVG the vector exchange copy, and PNG the
  preview. Embedded camera/PSF image fields are intentionally raster data.

## Evidence and limitations

`figure_manifest.json` records captions and evidence type. `data/` retains
curves, localizations, frame-wise simulation truth, run parameters, and image
arrays. The fixed-seed 600-frame two-source example is reconstructed by the
actual V1.2 centroid code; it enters as numerical frames, without video
encoding. Its 8-pixel separation is recovered as 8.016 pixels, with 0.456-pixel
2D position RMSE. These are synthetic position metrics, not experimental
resolution. The independent repeatability example uses 800 Poisson frames.

The manuscript's unsupported result placeholders were replaced by labelled
synthetic demonstrations. Physical apparatus performance still requires
traceable raw data, spatial calibration, and suitable uncertainty analysis.
Original experimental artwork and the preceding document/source versions
remain in `../../../archive/figure_revisions/` and the original asset tree.

The typography revision preserves every file in `data/` byte-for-byte.
`typography_audit.json` records point sizes, native widths, matched text spans
on the final pages, and compiled figure numbers. The preceding review package
is preserved in `../../../releases/` alongside the R2 package.

Physics references: [Thompson, Larson and Webb (2002)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1302065/),
[SMLM primer](https://www.nature.com/articles/s43586-021-00038-x),
[rendering-dependent resolution estimates](https://www.nature.com/articles/s42003-021-02086-1),
and [sampling and resolution](https://www.nature.com/articles/s41592-020-0963-0).
The Thompson background parameter is RMS noise in photon-equivalent units;
the plotted formula is a theoretical reference, not centroid uncertainty.
