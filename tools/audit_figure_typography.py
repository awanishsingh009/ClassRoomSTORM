"""Audit compiled figure sizes, font sizes, and LaTeX figure-number mappings.

Requires PyMuPDF. Run after build_figures.py, build_docs.py, and the manuscript
build. Mathematical subscripts/superscripts and original UI screenshots are
reported separately from normal scientific labels. Visual QA is still required.
"""
from pathlib import Path
import json
import re
import fitz
from build_figures import ROOT, SRC, PAPER, figure_locations
from project_paths import available_figures


def spans(page):
    return [s for b in page.get_text('dict')['blocks'] if 'lines' in b
            for line in b['lines'] for s in line['spans'] if s['text'].strip()]


def main():
    locations=figure_locations()
    manifest=available_figures(json.loads((SRC/'figure_manifest.json').read_text()))
    findings=[]
    errors=[]
    documents={}
    for row in manifest:
        name=row['name']
        raster=row.get('format')=='png'
        path=(PAPER if row['category']=='manuscript' else ROOT/'docs/figures'/row['category'])/(name+'.pdf')
        width_mm=None;labels=[]
        if not raster:
            with fitz.open(path) as pdf:
                page=pdf[0];width_mm=page.rect.width*25.4/72
                labels=spans(page)
        # TeX pt = 72/72.27 PDF points. Scripts naturally use smaller sizes.
        normal=[s for s in labels if s['size']>=8 or re.search(r'[A-Za-z]{4,}',s['text'])]
        normal_sizes=sorted({round(s['size']*72.27/72,2) for s in normal})
        if not raster and (not normal or any(not 8.48<=v<=9.02 for v in normal_sizes)):
            errors.append(f'{name}: unexpected ordinary label sizes {normal_sizes}')
        uses=[]
        for location in locations[name]:
            source=ROOT.parent/location['source']
            aux=(source.parent/'build'/(source.stem+'.aux')).read_text(errors='replace')
            match=re.search(r'\\newlabel\{'+re.escape(location['label'])+r'\}\{\{(\d+)\}\{(\d+)\}',aux)
            if not match or int(match[1])!=location['number']:
                errors.append(f'{source.name}: incorrect compiled number for {location["label"]}')
                continue
            if not raster and abs(width_mm-location['width_mm'])>.01:
                errors.append(f'{name}: native {width_mm:.3f} mm inserted at {location["width_mm"]} mm')
            target=source.with_suffix('.pdf')
            if target not in documents:documents[target]=fitz.open(target)
            page_index=int(match[2])-1
            printed=spans(documents[target][page_index])
            # Check shared, substantive text in the final page, not merely the
            # figure source. Resizing after inclusion must change these sizes.
            checked=0
            for s in normal:
                if len(s['text'].strip())<8:continue
                candidates=[t for t in printed if t['text']==s['text'] and t['font']==s['font']]
                if candidates:
                    delta=min(abs(t['size']-s['size']) for t in candidates)
                    if delta>.03:errors.append(f'{name}: printed scale changed for {s["text"]!r}')
                    checked+=1
            if not raster and checked<2:errors.append(f'{name}: insufficient matched labels on {source.stem} page {page_index+1}')
            uses.append(dict(document=location['document'],figure=int(match[1]),page=page_index+1,
                             inserted_width_mm=location['width_mm'],matched_label_spans=checked))
        findings.append(dict(master=name,native_width_mm=round(width_mm,3) if width_mm else None,
                             kind='original raster UI screenshot; embedded UI text excluded' if raster else 'scientific vector master',
                             normal_label_sizes_tex_pt=normal_sizes,
                             mathematical_script_spans=len(labels)-len(normal),locations=uses))
    for document in documents.values():document.close()
    report=dict(contract='9 pt main labels; 8.5 pt ticks, legends and notes; native-width inclusion',
                findings=findings,errors=errors)
    (SRC/'typography_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Audited {len(findings)} entries; {len(errors)} errors')
    for error in errors:print(error)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
