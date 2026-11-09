"""Build manuscript/access_overleaf.zip: a self-contained IEEE Access project that compiles on Overleaf.

Contents: access.tex (figure path rewritten to figures/), the official IEEE Access class files and logos,
sections/, tables/, figcaptions.tex, references.bib and the figures the paper uses."""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
FIG = ROOT / "results" / "figures"
OUT = MS / "access_overleaf.zip"

tex = (MS / "access.tex").read_text(encoding="utf8")
assert r"\graphicspath{{../results/figures/}}" in tex
tex = tex.replace(r"\graphicspath{{../results/figures/}}", r"\graphicspath{{figures/}}")

files = ["ieeeaccess.cls", "IEEEtran.cls", "Logo.png", "notaglineLogo.png", "bullet.png", "figcaptions.tex", "references.bib"]
inputs = re.findall(r"\\input\{([^}]+)\}", tex)
for p in (MS / "sections").glob("*.tex"):
    inputs += re.findall(r"\\input\{([^}]+)\}", p.read_text(encoding="utf8"))
texfiles = sorted({i if i.endswith(".tex") else i + ".tex" for i in inputs} - set(files))

figs = set()
for p in [MS / "access.tex", MS / "figcaptions.tex", *(MS / "sections").glob("*.tex")]:
    figs |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", p.read_text(encoding="utf8")))
figs = sorted(f for f in figs if not f.endswith(".png") or (FIG / f).exists())

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("access.tex", tex)
    for f in files + texfiles:
        z.write(MS / f, f)
    for f in figs:
        z.write(FIG / f, "figures/" + f)
    n = len(z.namelist())
print(OUT, "files:", n, "figures:", len(figs))
