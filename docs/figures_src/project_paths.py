"""Paths shared by figure tools in a standalone clone or the research workspace."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'docs/figures_src'
MANUSCRIPT = ROOT.parent / 'manuscript_writing'
PAPER = MANUSCRIPT / 'figures'
HAS_MANUSCRIPT = (MANUSCRIPT / 'ClassRoomSTORM_AJP.tex').is_file()


def document_sources():
    sources = [
        ('Theory notes', ROOT / 'docs/latex/ClassRoomSTORM_Theory.tex'),
        ('User guide', ROOT / 'docs/latex/ClassRoomSTORM_Help.tex'),
    ]
    if HAS_MANUSCRIPT:
        sources.append(('Manuscript', MANUSCRIPT / 'ClassRoomSTORM_AJP.tex'))
    return sources


def available_figures(manifest):
    """A standalone software clone builds and audits the two supplied manuals."""
    return [row for row in manifest if HAS_MANUSCRIPT or row['category'] != 'manuscript']
