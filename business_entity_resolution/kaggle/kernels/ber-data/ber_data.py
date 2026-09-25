# Downloads the organiser-provided dataset zip once; this kernel's output is attached as
# input ("ber-data") by every other pipeline kernel.
import os
import shutil
import subprocess
import zipfile

subprocess.run(["pip", "install", "-q", "gdown"], check=True)
import gdown  # noqa: E402

gdown.download(id="1-bH1Jp73PCwp7qFPZVOJ4gscSMhbTNR2", output="/tmp/data.zip", quiet=False)
zipfile.ZipFile("/tmp/data.zip").extractall("/tmp/x")
src = "/tmp/x/student_resource"
for sub in ("dataset", "utils"):
    shutil.copytree(f"{src}/{sub}", f"/kaggle/working/{sub}", ignore=shutil.ignore_patterns(".DS_Store"))
shutil.copy(f"{src}/Documentation_template.md", "/kaggle/working/")
for root, _, files in os.walk("/kaggle/working"):
    for f in files:
        p = os.path.join(root, f)
        print(p, os.path.getsize(p))
