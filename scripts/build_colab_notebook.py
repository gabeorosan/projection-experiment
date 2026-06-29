"""Generate a self-contained Colab notebook from the projexp modules.

The notebook recreates the projexp/ package on the Colab filesystem via %%writefile
cells (so the code is byte-identical to this repo), installs deps, and runs the full
gen -> train -> eval -> analyze pipeline. Re-run this script whenever the modules change.

    python3 scripts/build_colab_notebook.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "projexp")
OUT = os.path.join(ROOT, "experiments", "legacy_colab", "projection_experiment_colab.ipynb")

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODULES = ["__init__", "items", "prompts", "scoring", "modeling", "train", "eval", "analyze"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def writefile_cell(module: str) -> dict:
    with open(os.path.join(SRC, f"{module}.py")) as f:
        body = f.read()
    return code(f"%%writefile projexp/{module}.py\n{body}")


cells = [
    md(
        "# Projection fine-tuning experiment\n"
        "\n"
        "Does fine-tuning a model to be **risk-seeking** vs **risk-averse** change what it "
        "thinks **other people** would do? (Social projection installed by fine-tuning.)\n"
        "\n"
        "## How to run (3 clicks)\n"
        "1. **Runtime → Change runtime type → T4 GPU → Save.**\n"
        "2. **Runtime → Run all.**\n"
        "3. Wait ~30–60 min. The final cell prints the projection result.\n"
        "\n"
        "You don't need to edit anything. If a cell ever errors about package versions, do "
        "**Runtime → Restart session**, then Run all again.\n"
    ),
    code("!nvidia-smi -L   # confirms a GPU is attached; if this errors, set the GPU runtime (step 1)"),
    code("!pip install -q -U transformers peft accelerate datasets bitsandbytes"),
    md("### Recreate the `projexp` package on this machine\n"
       "(These cells just write the code to files — nothing to change.)"),
    code("import os; os.makedirs('projexp', exist_ok=True)"),
]

cells += [writefile_cell(m) for m in MODULES]

cells += [
    md("### 1. Generate data (first-person training sets + shared eval set)"),
    code("!python -m projexp.gen_data --out data"),
    md("### 2. Fine-tune the three arms (QLoRA, ~5–10 min each on a T4)"),
    code(f"!python -m projexp.train --model {MODEL} --data data/arm_risk_seeking.jsonl "
         f"--out runs/risk_seeking --load-4bit --max-steps 200"),
    code(f"!python -m projexp.train --model {MODEL} --data data/arm_risk_averse.jsonl "
         f"--out runs/risk_averse --load-4bit --max-steps 200"),
    code(f"!python -m projexp.train --model {MODEL} --data data/arm_control_ev.jsonl "
         f"--out runs/control_ev --load-4bit --max-steps 200"),
    md("### 3. Evaluate baseline + the three fine-tuned arms"),
    code(f"!python -m projexp.eval --model {MODEL} --arm baseline "
         f"--items data/eval.jsonl --out results/baseline.json --load-4bit --numeric"),
    code(f"!python -m projexp.eval --model {MODEL} --adapter runs/risk_seeking --arm risk_seeking "
         f"--items data/eval.jsonl --out results/risk_seeking.json --load-4bit --numeric"),
    code(f"!python -m projexp.eval --model {MODEL} --adapter runs/risk_averse --arm risk_averse "
         f"--items data/eval.jsonl --out results/risk_averse.json --load-4bit --numeric"),
    code(f"!python -m projexp.eval --model {MODEL} --adapter runs/control_ev --arm control_ev "
         f"--items data/eval.jsonl --out results/control_ev.json --load-4bit --numeric"),
    md("### 4. The result\n"
       "Read it top to bottom: the **own-choice delta** must be clearly positive (the trait "
       "installed), then the **IMPLICIT others delta** with its 95% CI is the projection effect "
       "— if the CI is above 0, inducing the trait shifted the model's beliefs about other "
       "people in the same direction."),
    code("!python -m projexp.analyze "
         "--results results/baseline.json results/risk_seeking.json "
         "results/risk_averse.json results/control_ev.json --out results/summary.json"),
    md("### (optional) Download the raw results\n"
       "Run this to save the JSON to your computer."),
    code("from google.colab import files\n"
         "import shutil\n"
         "shutil.make_archive('projexp_results', 'zip', 'results')\n"
         "files.download('projexp_results.zip')"),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
