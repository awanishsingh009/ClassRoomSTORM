"""Rebuild editable scientific figures and a captioned review gallery."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'docs/figures_src'))
from project_paths import ROOT, SRC, PAPER, HAS_MANUSCRIPT, document_sources, available_figures
PAPER_MASTERS={'P1_airy_gaussian':'T1_airy_gaussian','P2_rayleigh':'T2_rayleigh','P4_image_formation':'T3_image_formation','P5_precision':'T6_precision_vs_photons','P7_pipeline':'T10_pipeline','P8_synthetic_result':'T11_raw_vs_reconstruction','P9_repeatability':'T5_poisson_noise'}

TITLES={
    'T1_airy_gaussian':'Airy PSF and Gaussian approximation',
    'T2_rayleigh':'Two-point Rayleigh criterion',
    'T3_image_formation':'Incoherent image formation',
    'T4_blinking':'Temporal sparsity',
    'T5_poisson_noise':'Noise and repeated position estimates',
    'T6_precision_vs_photons':'Theoretical localization precision',
    'T7_centroid':'Thresholded intensity centroid',
    'T8_precision_accuracy':'Spread and bias',
    'T9_nyquist':'Spatial sampling',
    'T10_pipeline':'Reconstruction workflow',
    'T11_raw_vs_reconstruction':'Synthetic reconstruction',
    'H1_concept_loop':'Localization imaging concept',
    'H2_launcher_screenshot':'Studio launcher',
    'H3_quickstart_montage':'Quick-start windows',
    'H4_pattern_preview':'Source pattern',
    'H5_blinking_frame':'Simulated camera frame',
    'H6_reconstruction_load_crop':'Load and crop interface',
    'H7_results_browser':'Results interface',
    'H8_linked_profile':'Linked image profiles',
    'H9_output_map':'Output files',
    'P3_model_mismatch':'Single-Gaussian model mismatch',
    'P6_apparatus':'Apparatus and optical pathway',
    'P10_position_validation':'Position validation against truth',
    'P11_rendering_sensitivity':'Effect of display-kernel width',
}


def run(args,cwd):
    p=subprocess.run([str(a) for a in args],cwd=cwd,capture_output=True,text=True,errors='replace')
    if p.returncode:raise RuntimeError(p.stdout[-4500:]+p.stderr[-1500:])
    if p.stderr.strip():print(p.stderr[-500:])
    return p.stdout


def tex(source,dest):
    build=SRC/'build';build.mkdir(exist_ok=True)
    for _ in range(2):run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={build}',source.name],source.parent)
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(build/(source.stem+'.pdf'),dest)
    run(['pdftoppm','-singlefile','-r','300','-png',dest,dest.with_suffix('')],ROOT)
    run(['pdftocairo','-svg',dest,dest.with_suffix('.svg')],ROOT)
    log=(build/(source.stem+'.log')).read_text(errors='replace')
    warnings=[x for x in log.splitlines() if 'Overfull' in x or 'Warning:' in x]
    print(source.name,'warnings:',warnings)


def escape(t):
    for old,new in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#'),('^',r'\textasciicircum{}')]:t=t.replace(old,new)
    return t


def figure_locations():
    """Read document order, labels and insertion widths, never asset-ID numbers."""
    locations={}
    for title,source in document_sources():
        content=re.sub(r'(?m)^\s*%.*$', '', source.read_text(encoding='utf-8'))
        blocks=re.findall(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',content,re.S)
        for number,block in enumerate(blocks,1):
            image=re.search(r'\\includegraphics\[width=([^\]]+)\]\{([^}]+)\}',block)
            label=re.search(r'\\label\{([^}]+)\}',block)
            if not image or not label:
                raise ValueError(f'Figure {number} in {source.name} needs one image, width and label')
            width,asset=image.groups();asset=Path(asset).stem
            master=PAPER_MASTERS.get(asset,asset)
            if width.endswith('mm'):width_mm=float(width[:-2])
            elif width.endswith(r'\linewidth'):width_mm=162*float(width[:-10] or 1)
            else:raise ValueError(f'Unsupported figure width: {width}')
            locations.setdefault(master,[]).append(dict(document=title,number=number,
                label=label.group(1),source=source.relative_to(ROOT.parent).as_posix(),
                asset=asset,width_mm=width_mm))
    return locations


def caption_tex(text):
    text=escape(text)
    for old,new in [
        ('0.610 lambda/NA',r'$0.610\,\lambda/\mathrm{NA}$'),
        ('b\\_rms',r'$b_{\mathrm{rms}}$'),
        ('N\\textasciicircum{}(-1/2)',r'$N^{-1/2}$'),
        ('N\\textasciicircum{}(-1)',r'$N^{-1}$'),
        ('PSF sigma=1.5',r'PSF $\sigma=1.5$'),
        ('pixel pitch a=1',r'pixel pitch $a=1$'),
        ('sigma=2.3',r'$\sigma=2.3$'),
        ('q=0.98',r'$q=0.98$'),
        ('q=0.995',r'$q=0.995$'),
        ('Gaussian sigma 6',r'Gaussian $\sigma=6$'),
        ('rendering sigma=1.5',r'rendering $\sigma_r=1.5$'),
        ('Gaussian sigma 0.75, 1.5 or 3',r'Gaussian $\sigma_r=0.75$, $1.5$ or $3$'),
    ]:text=text.replace(old,new)
    return text


def build_gallery(manifest):
    manifest=available_figures(manifest)
    locations=figure_locations()
    by_name={row['name']:row for row in manifest}
    if set(locations)!=set(by_name):
        raise ValueError(f'Gallery/document mismatch: {set(locations)^set(by_name)}')
    pages=[]
    for name,uses in locations.items():
        row=by_name[name];primary=uses[0]
        p=(PAPER if row['category']=='manuscript' else ROOT/'docs/figures'/row['category'])/(name+'.'+row.get('format','pdf'))
        where='; '.join(f"{u['document']}, Figure {u['number']}" for u in uses)
        heading=f"{primary['document']}: Figure {primary['number']}"
        width=primary['width_mm']
        # Height cap is only for original UI screenshots; scientific masters
        # are always inserted at their explicitly specified physical width.
        size=f'width={width:.2f}mm'+(r',height=.65\textheight,keepaspectratio' if row.get('format')=='png' else '')
        pages.append(r'\textbf{'+escape(heading)+r'}\par\smallskip '+escape(TITLES[name])+
            r'\par\medskip\begin{center}\includegraphics['+size+r']{'+p.as_posix()+
            r'}\end{center}{\small\textbf{Caption.} '+caption_tex(row['caption'])+
            r'\par\medskip\textbf{Document locations.} '+escape(where)+
            r'\par\smallskip\textbf{Evidence type.} '+escape(row['scope'])+r'\par}\clearpage')
    (SRC/'figure_locations.json').write_text(json.dumps(locations,indent=2)+'\n')
    gallery=SRC/'Scientific_Figure_Review.tex'
    gallery.write_text(r'\documentclass[11pt,a4paper]{article}\usepackage[T1]{fontenc}\usepackage{lmodern,graphicx,geometry}\geometry{margin=24mm}\setlength{\parindent}{0pt}\begin{document}'+ '\n'.join(pages)+r'\end{document}')
    tex(gallery,ROOT/'docs/Scientific_Figure_Review.pdf')


def main():
    if '--gallery-only' in sys.argv:
        build_gallery(json.loads((SRC/'figure_manifest.json').read_text()))
        return
    env=dict(os.environ);env['PYTHONIOENCODING']='utf-8'
    p=subprocess.run([sys.executable,str(SRC/'make_scientific_figures.py')],cwd=ROOT,env=env)
    if p.returncode:raise SystemExit(p.returncode)
    manifest=json.loads((SRC/'figure_manifest.json').read_text())
    diagrams=[('T10_pipeline','theory','The current centroid pipeline, with a separate raw-mean branch. Rendering width is fixed and is not an uncertainty estimate.'),
              ('H1_concept_loop','help','Conceptual progression from sparse blinking through position estimation to comparison with data or known truth.'),
              ('H9_output_map','help','Output roles for the virtual experiment and reconstruction workflows.'),
              ('P6_apparatus','manuscript','Supplied apparatus CAD image and a conceptual optical/data pathway. Dimensions and effective blur must be calibrated; no raw-camera inset is claimed.')]
    for name,kind,caption in diagrams:
        if kind=='manuscript' and not HAS_MANUSCRIPT:
            continue
        dest=(PAPER if kind=='manuscript' else ROOT/'docs/figures'/kind)/(name+'.pdf')
        tex(SRC/'tikz'/(name+'.tex'),dest)
        manifest.append(dict(name=name,category=kind,caption=caption,scope='native TikZ explanatory schematic'))
    screenshots = {
        'H2_launcher_screenshot':'Current Studio launcher, captured from the application.',
        'H3_quickstart_montage':'Current virtual-experiment and reconstruction windows, captured from the application.',
        'H6_reconstruction_load_crop':'Current reconstruction load/crop interface before a video is loaded.',
        'H7_results_browser':'Current results browser before a reconstruction is run. This is a UI screenshot, not experimental evidence.',
    }
    for name, caption in screenshots.items():
        if (ROOT/'docs/figures/help'/(name+'.png')).exists():
            manifest.append(dict(name=name,category='help',caption=caption,scope='actual application screenshot',format='png'))
    # Publish manuscript copies from the same reviewed masters.
    for name,original in PAPER_MASTERS.items():
        if not HAS_MANUSCRIPT:
            break
        for ext in ['pdf','png','svg']:shutil.copy2(ROOT/'docs/figures/theory'/(original+'.'+ext),PAPER/(name+'.'+ext))
    (SRC/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    build_gallery(manifest)
    suffix=' plus manuscript derivatives' if HAS_MANUSCRIPT else ' (standalone manuals)'
    print('Built',len(manifest),'scientific figures and UI review entries plus gallery'+suffix+'.')


if __name__=='__main__':main()
