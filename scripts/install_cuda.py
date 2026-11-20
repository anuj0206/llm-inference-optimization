"""Install a user-local CUDA toolkit from NVIDIA's hash-verified redistributables."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
VERSION = "13.5.0"
ROOT = Path(os.environ["LOCALAPPDATA"]) / "LLMOptimization"
TOOLKIT = ROOT / "cuda" / VERSION
COMPONENTS = (
    "cccl", "cuda_crt", "cuda_cudart", "cuda_nvcc", "libnvvm",
    "libcublas", "cuda_cuobjdump", "cuda_nvdisasm", "cuda_profiler_api", "cuda_nvml_dev",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_component(item):
    name, entry = item
    destination_root = (ROOT / "tools" / name) if name.startswith("nsight_") else TOOLKIT
    stamp = destination_root / ".components" / f"{name}.sha256"
    if stamp.is_file() and stamp.read_text().strip() == entry["sha256"]:
        print(f"Already installed {name}", flush=True)
        return
    cache = ROOT / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / Path(entry["relative_path"]).name
    if not archive.exists() or sha256(archive) != entry["sha256"]:
        print(f"Downloading {name}: {int(entry['size']) / 1024**2:.1f} MiB", flush=True)
        partial = archive.with_suffix(".partial")
        subprocess.run([
            "curl.exe", "--fail", "--location", "--silent", "--show-error", "--retry", "4",
            "--connect-timeout", "30", "--speed-time", "60", "--speed-limit", "1024",
            "--continue-at", "-", "--output", str(partial), BASE + entry["relative_path"],
        ], check=True)
        if sha256(partial) != entry["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {name}")
        partial.replace(archive)
    # NVIDIA component archives contain one top-level directory. Merge their
    # bin/include/lib layout, retaining each component's license separately.
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            relative = Path(*Path(member.filename).parts[1:])
            if not relative.parts or member.is_dir():
                continue
            destination = destination_root / relative
            if ".." in relative.parts or relative.is_absolute():
                raise RuntimeError("Unsafe archive path")
            if len(relative.parts) == 1 and relative.suffix.lower() in (".txt", ".pdf"):
                destination = destination_root / "licenses" / name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(entry["sha256"] + "\n")
    print(f"Installed and verified {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profilers", action="store_true", help="Also install Nsight Compute and Systems")
    parser.add_argument("--components", nargs="+", help="Install only the named manifest components")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    TOOLKIT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE + f"redistrib_{VERSION}.json", timeout=60) as response:
        manifest = json.load(response)
    names = args.components or (COMPONENTS + (("nsight_compute", "nsight_systems") if args.profilers else ()))
    entries = [(name, manifest[name]["windows-x86_64"]) for name in names]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(install_component, entries))
    (TOOLKIT / "redistrib-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (TOOLKIT / "version.json").write_text(json.dumps({"cuda": {"version": VERSION}}), encoding="utf-8")
    print(f"CUDA toolkit ready at {TOOLKIT}", flush=True)
