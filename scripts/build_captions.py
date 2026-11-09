"""Turn results/figures/captions.json into manuscript/figcaptions.tex (\\CAP{<figure name>})."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cap = json.loads((ROOT / "results/figures/captions.json").read_text(encoding="utf8"))
REPL = [("\\", r"\textbackslash{}"), ("%", r"\%"), ("&", r"\&"), ("_", r"\_"), ("#", r"\#"),
        ("θ", r"$\theta$"), ("φ", r"$\phi$"), ("τ", r"$\tau$"), ("Δ", r"$\Delta$"), ("≥", r"$\geq$"),
        ("≤", r"$\leq$"), ("×", r"$\times$"), ("−", "-"), ("±", r"$\pm$"), ("≈", r"$\approx$"), ("β", r"$\beta$"), ("κ", r"$\kappa$")]
lines = [r"\newcommand{\CAP}[1]{\csname cap@#1\endcsname}"]
for k, v in cap.items():
    for a, b in REPL:
        v = v.replace(a, b)
    lines.append(r"\expandafter\def\csname cap@" + k + r"\endcsname{" + v + "}")
(ROOT / "manuscript/figcaptions.tex").parent.mkdir(parents=True, exist_ok=True)   # manuscript/ is not in the repository
(ROOT / "manuscript/figcaptions.tex").write_text("\n".join(lines) + "\n", encoding="utf8")
print(len(cap), "captions")
