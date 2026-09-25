"""Generate self-contained Kaggle notebooks that embed the current src/ber package.

  A_cpu_pipeline.ipynb : data download -> prepare -> re-ranker -> candidates -> context
                         (CPU session; its /kaggle/working output feeds notebook B)
  B_gpu_match.ipynb    : cross-encoder train/score on 2xT4 -> matcher -> submission

Run: python business_entity_resolution/kaggle/build_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src" / "ber"
GDRIVE_ID = "1-bH1Jp73PCwp7qFPZVOJ4gscSMhbTNR2"  # organiser-provided dataset zip
PIP = "!pip install -q polars==1.44.2 rapidfuzz==3.14.6 anyascii==0.3.3 lightgbm==4.7.0 gdown"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def src_cells(root: str):
    cells = [code(f"import os\nos.makedirs('{root}/ber', exist_ok=True)")]
    for f in sorted(SRC.glob("*.py")):
        cells.append(code(f"%%writefile {root}/ber/{f.name}\n" + f.read_text()))
    return cells


def env(work: str, data: str, out: str, root: str) -> str:
    return (
        "import os\n"
        f"os.environ['BER_WORK'] = '{work}'\n"
        f"os.environ['BER_DATA'] = '{data}'\n"
        f"os.environ['BER_OUTPUT'] = '{out}'\n"
        f"os.environ['PYTHONPATH'] = '{root}'\n"
        f"os.makedirs('{work}', exist_ok=True)\n"
    )


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_a():
    root, work = "/kaggle/working/code", "/kaggle/working/work"
    data = "/tmp/ber/student_resource/dataset"
    cells = [
        md("# A · CPU pipeline\nSettings: **Internet ON**, accelerator **None**. "
           "Save version with *Save & Run All*; the `/kaggle/working/work` output feeds notebook B."),
        code(PIP),
        code(
            "import gdown, zipfile, os\n"
            "os.makedirs('/tmp/ber', exist_ok=True)\n"
            f"gdown.download(id='{GDRIVE_ID}', output='/tmp/ber/data.zip', quiet=False)\n"
            "zipfile.ZipFile('/tmp/ber/data.zip').extractall('/tmp/ber')\n"
            "!find /tmp/ber -maxdepth 3 -name '*.tsv' | head; rm /tmp/ber/data.zip"
        ),
        *src_cells(root),
        code(env(work, data, "/kaggle/working/output", root)),
        code("!python -m ber.prepare --splits train test"),
        code("!python -m ber.rerank_dev --n-train 40000 --n-eval 20000"),
        code("!python -m ber.candidates --splits train test --super-chunk 400000 --chunk 10000"),
        code("!python -m ber.matcher ctx --split train\n!python -m ber.matcher ctx --split test"),
        code("# keep only what notebook B needs\n"
             "!cd /kaggle/working/work && rm -f *_keys.parquet *_source*.parquet dev_feats_*.parquet && du -sh . && ls -la"),
    ]
    return notebook(cells)


def build_b():
    root, work = "/kaggle/working/code", "/kaggle/working/work"
    cells = [
        md("# B · GPU cross-encoder + matcher\nSettings: **Internet ON**, accelerator **GPU T4 x2**. "
           "Add notebook A's output as input (Add Input → Notebooks) and set `A_OUT` below."),
        code(PIP + " accelerate"),
        code("A_OUT = '/kaggle/input/a-cpu-pipeline/work'  # adjust to the attached input path\n"
             f"!mkdir -p {work} && cp -r $A_OUT/. {work}/"),
        code(
            "import gdown, zipfile, os\n"
            "os.makedirs('/tmp/ber', exist_ok=True)\n"
            f"gdown.download(id='{GDRIVE_ID}', output='/tmp/ber/data.zip', quiet=False)\n"
            "zipfile.ZipFile('/tmp/ber/data.zip').extractall('/tmp/ber'); os.remove('/tmp/ber/data.zip')"
        ),
        *src_cells(root),
        code(env(work, "/tmp/ber/student_resource/dataset", "/kaggle/working/output", root)),
        code("!accelerate launch --multi_gpu --num_processes 2 -m ber.cross_encoder train --keep-k 20 --neg-per-q 8 --epochs 1 --bs 64"),
        code("!accelerate launch --multi_gpu --num_processes 2 -m ber.cross_encoder score --split train --keep-k 10 --bs 256 --share 0.3\n"
             "!accelerate launch --multi_gpu --num_processes 2 -m ber.cross_encoder score --split test --keep-k 10 --bs 256"),
        code("!python -m ber.matcher oof\n!python -m ber.matcher final"),
        code("!ls -la /kaggle/working/output && head -3 /kaggle/working/output/matching_results.tsv"),
    ]
    return notebook(cells)


if __name__ == "__main__":
    for name, nb in (("A_cpu_pipeline.ipynb", build_a()), ("B_gpu_match.ipynb", build_b())):
        (HERE / name).write_text(json.dumps(nb, indent=1))
        print("wrote", HERE / name)
