"""Compile both manuals from their source directory and publish successful PDFs."""
from pathlib import Path
import os
import shutil
import subprocess


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo / "docs/latex"
    output = source / "build"
    output.mkdir(exist_ok=True)
    engine = shutil.which("pdflatex")
    if not engine:
        raise SystemExit("pdflatex is required to build the manuals")
    env = dict(os.environ)
    # Run inside the build directory so stale auxiliary files next to the
    # sources cannot shadow the current table of contents or cross-references.
    env["TEXINPUTS"] = os.pathsep.join(str(repo / "docs/figures" / kind) + "//" for kind in ("help", "theory")) + os.pathsep
    for name in ("ClassRoomSTORM_Help", "ClassRoomSTORM_Theory"):
        previous = None
        for _ in range(5):
            run = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", str(source / (name + ".tex"))], cwd=output, env=env, capture_output=True, text=True, errors="replace")
            if run.returncode:
                raise SystemExit(run.stdout[-6000:] + run.stderr[-1000:])
            state = tuple((output / (name + suffix)).read_bytes() for suffix in (".aux", ".toc", ".out"))
            if state == previous:
                break
            previous = state
        else:
            raise SystemExit(f"{name}: references did not converge in five passes")
        log = (output / (name + ".log")).read_text(errors="replace")
        warnings = [line for line in log.splitlines() if "Warning:" in line or "Overfull" in line]
        for dest in (source, repo / "docs"):
            shutil.copy2(output / (name + ".pdf"), dest / (name + ".pdf"))
        print(f"{name}: built; warnings: {warnings}")


if __name__ == "__main__":
    main()
