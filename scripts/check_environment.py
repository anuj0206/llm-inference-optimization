"""Inspect local tools without network access or third-party dependencies."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def run_local(args):
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=30, errors="replace"
        )
        return {"exit_code": result.returncode,
                "output": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"exit_code": None, "output": str(exc)}


def inspect():
    tools = {name: shutil.which(name) for name in (
        "git", "python", "uv", "cmake", "ninja", "cl", "nvcc", "nsys", "ncu"
    )}
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    visual_studio = []
    if vswhere.is_file():
        found = run_local([str(vswhere), "-all", "-products", "*", "-format", "json"])
        if found["exit_code"] == 0:
            for instance in json.loads(found["output"]):
                root = Path(instance["installationPath"])
                visual_studio.append({
                    "name": instance.get("displayName"),
                    "path": str(root),
                    "compilers": [str(p) for p in root.glob("VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe")],
                    "cmake": [str(p) for p in root.glob("Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe")],
                    "ninja": [str(p) for p in root.glob("Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe")],
                })
    cuda_root = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA GPU Computing Toolkit/CUDA"
    gpu_tool = shutil.which("nvidia-smi")
    gpu = run_local([gpu_tool, "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv"]) if gpu_tool else None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "virtual_environment": sys.prefix != sys.base_prefix,
        "tools_on_path": tools,
        "visual_studio": visual_studio,
        "cuda_compilers": [str(p) for p in cuda_root.glob("*/bin/nvcc.exe")],
        "gpu": gpu,
        "note": "Inventory only; compiler compatibility and CUDA builds are not yet verified.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(inspect(), indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
