"""Push a pipeline job to Kaggle as a private script kernel and (optionally) wait for it.

  python business_entity_resolution/kaggle/push_job.py ber-prep \
      --steps "ber.prepare --splits train test" --pip "sparse_dot_topn"
  python .../push_job.py ber-rerank --after ber-prep --steps "ber.rerank_dev" --wait

Every job attaches: ber-code (dataset), ber-data (kernel output) and the outputs of the
kernels listed in --after, whose work/ files are symlinked into this job's work dir.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

USER = "peeyushprashant"
ROOT = Path(__file__).resolve().parents[2]
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
JOBS = ROOT / "work" / "kaggle_jobs"
PIP = "polars==1.44.2 rapidfuzz==3.14.6 anyascii==0.3.3 lightgbm==4.7.0 sparse_dot_topn"

RUNNER = r'''
import glob, os, subprocess, sys, time
STEPS = {steps!r}
subprocess.run("pip install -q {pip}", shell=True, check=True)
subprocess.run("find -L /kaggle/input -maxdepth 4 | grep -v '/dataset/.*tsv$' | head -60", shell=True)


def find_dir(name, parent=None):
    for root, _, files in os.walk("/kaggle/input", followlinks=True):
        if name in files and (parent is None or os.path.basename(root) == parent):
            return root
    return None


# Kaggle flattens the uploaded code zip, so rebuild the `ber` package with a symlink.
src = find_dir("matcher.py")
code = "/kaggle/working/code"
os.makedirs(code, exist_ok=True)
if not os.path.exists(f"{{code}}/ber"):
    os.symlink(src, f"{{code}}/ber")
data = os.path.dirname(find_dir("train_source1.tsv", "train"))
work = "/kaggle/working/work"
os.makedirs(work, exist_ok=True)
links = []
for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
    if os.path.basename(root) != "work":
        continue
    for fn in files + dirs:  # dirs too, e.g. work/ce_model
        dst = os.path.join(work, fn)
        if not os.path.exists(dst):
            os.symlink(os.path.join(root, fn), dst); links.append(dst)
env = dict(os.environ, PYTHONPATH=code, BER_DATA=data, BER_WORK=work,
           BER_OUTPUT="/kaggle/working/output", POLARS_MAX_THREADS="4", PYTHONUNBUFFERED="1")
# GPU jobs: steps may use $ACCEL = accelerate launch sized to the visible GPUs.
ngpu = int(subprocess.run("nvidia-smi -L 2>/dev/null | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 0)
env["ACCEL"] = ("accelerate launch --mixed_precision fp16 --num_machines 1 --dynamo_backend no "
                + (f"--multi_gpu --num_processes {{ngpu}}" if ngpu > 1 else "--num_processes 1") + " -m")
subprocess.run("nvidia-smi -L 2>/dev/null; nproc; free -g | head -2", shell=True)
print("code", code, "| data", data, "| linked", len(links), "files | GPUs", ngpu, flush=True)
ok = True
for s in STEPS:
    t = time.time()
    print(f"\n===== {{s}}", flush=True)
    cmd = s[1:] if s.startswith("!") else f"{{sys.executable}} -m {{s}}"
    r = subprocess.run(cmd, shell=True, env=env)
    print(f"===== {{s}} -> exit {{r.returncode}} in {{time.time()-t:.0f}}s", flush=True)
    if r.returncode != 0:
        ok = False
        break
for d in links + [f"{{code}}/ber"]:
    os.unlink(d)
subprocess.run("du -sh /kaggle/working/*; ls -la /kaggle/working/work | head -50", shell=True)
sys.exit(0 if ok else 1)
'''


def push(name: str, steps: list[str], after: list[str], gpu: bool, pip: str) -> None:
    d = JOBS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.py").write_text(RUNNER.format(steps=steps, pip=pip))
    meta = {
        "id": f"{USER}/{name}", "title": name, "code_file": "job.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
        "dataset_sources": [f"{USER}/ber-code"],
        "kernel_sources": [f"{USER}/ber-data"] + [f"{USER}/{a}" for a in after],
        "competition_sources": [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    subprocess.run([KAGGLE, "kernels", "push", "-p", str(d)], check=True)


def status(name: str) -> str:
    r = subprocess.run([KAGGLE, "kernels", "status", f"{USER}/{name}"], capture_output=True, text=True)
    return r.stdout.strip()


def fetch_log(name: str) -> Path:
    d = JOBS / name / "out"
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([KAGGLE, "kernels", "output", f"{USER}/{name}", "-p", str(d), "--file-pattern", r".*\.log$"],
                   capture_output=True, text=True)
    return d


def read_log(name: str) -> str:
    d = fetch_log(name)
    logs = sorted(d.glob("*.log"))
    if not logs:
        return "(no log yet)"
    return "".join(e.get("data", "") for e in json.loads(logs[0].read_text()))


def wait(name: str, poll: int = 60) -> str:
    while True:
        s = status(name)
        if any(x in s for x in ("COMPLETE", "ERROR", "CANCEL")):
            return s
        time.sleep(poll)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--steps", nargs="+", default=[])
    ap.add_argument("--after", nargs="*", default=[])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--pip", default=PIP)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--log", action="store_true", help="only print the finished job's log")
    a = ap.parse_args()
    if a.log:
        print(read_log(a.name))
        sys.exit(0)
    push(a.name, a.steps, a.after, a.gpu, a.pip)
    if a.wait:
        print(wait(a.name))
        print(read_log(a.name)[-6000:])
        sys.exit(0)
